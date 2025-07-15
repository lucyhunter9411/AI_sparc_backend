"""
Azure Blob Storage Service

This service provides a unified interface for Azure Blob Storage operations,
encapsulating all blob storage access and exposing generic file functions.
"""

import os
import io
import uuid
import logging
from typing import List, Dict, Optional, Union, BinaryIO
from pathlib import Path
from datetime import datetime, timedelta

from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
    BlobSasPermissions
)
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError, ServiceRequestError
from PIL import Image

logger = logging.getLogger(__name__)

class BlobStorageService:
    """
    Service class for Azure Blob Storage operations.
    
    Provides a unified interface for blob storage operations including:
    - File upload/download
    - Directory listing
    - File deletion
    - Image conversion and optimization
    - URL generation
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the blob storage service.
        
        Args:
            connection_string: Azure Storage connection string. 
                             If None, reads from AZURE_STORAGE_CONNECTION_STRING env var.
        """
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("Azure Storage connection string is required")

        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.account_name = self.blob_service_client.account_name

    def get_container_client(self, container_name: str):
        """Get a container client for the specified container."""
        return self.blob_service_client.get_container_client(container_name)

    def get_blob_client(self, container_name: str, blob_path: str):
        """Get a blob client for the specified container and blob path."""
        return self.get_container_client(container_name).get_blob_client(blob_path)

    def generate_blob_url(self, container_name: str, blob_path: str) -> str:
        """Generate a public blob URL."""
        return f"https://{self.account_name}.blob.core.windows.net/{container_name}/{blob_path}"

    def generate_sas_url(self, container_name: str, blob_path: str, expiry_minutes: int = 60) -> str:
        """Generate a Shared Access Signature URL."""
        sas_token = generate_blob_sas(
            account_name=self.account_name,
            container_name=container_name,
            blob_name=blob_path,
            account_key=self.blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(minutes=expiry_minutes)
        )
        return f"{self.generate_blob_url(container_name, blob_path)}?{sas_token}"

    def put(self, container_name: str, blob_path: str, data: Union[bytes, BinaryIO], 
            content_type: Optional[str] = None, overwrite: bool = True) -> str:
        """
        Upload data to blob storage.
        
        Args:
            container_name: Name of the container
            blob_path: Path within the container
            data: Data to upload (bytes or file-like object)
            content_type: MIME type of the data
            overwrite: Whether to overwrite existing blob
            
        Returns:
            Public URL of the uploaded blob
        """
        try:
            blob_client = self.get_blob_client(container_name, blob_path)

            content_settings = ContentSettings(content_type=content_type) if content_type else None

            blob_client.upload_blob(
                data,
                overwrite=overwrite,
                content_settings=content_settings
            )

            blob_url = self.generate_blob_url(container_name, blob_path)
            logger.info(f"✅ Successfully uploaded blob: {blob_path} to {blob_url}")
            return blob_url
        
        except Exception as e:
            logger.error(f"❌ Failed to upload blob {blob_path}: {e}")
            raise

    def get(self, container_name: str, blob_path: str) -> bytes:
        """
        Download data from blob storage.
        
        Args:
            container_name: Name of the container
            blob_path: Path within the container
            
        Returns:
            Blob data as bytes
        """
        try:
            blob_client = self.get_blob_client(container_name, blob_path)
            blob_data = blob_client.download_blob().readall()
            logger.info(f"✅ Successfully downloaded blob: {blob_path}")
            return blob_data
        
        except ResourceNotFoundError:
            logger.error(f"❌ Blob not found: {blob_path}")
            raise
        except ServiceRequestError as e:
            logger.error(f"❌ Timeout while downloading blob: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            raise

    def ls(self, container_name: str, prefix: str = "") -> List[Dict[str, str]]:
        """
        List blobs in a container with optional prefix.
        
        Args:
            container_name: Name of the container
            prefix: Optional prefix to filter blobs
            
        Returns:
            List of blob information dictionaries
        """
        try:
            container_client = self.get_container_client(container_name)
            blobs = []
            
            for blob in container_client.list_blobs(name_starts_with=prefix):
                blobs.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                    "url": self.generate_blob_url(container_name, blob.name)
                })
            
            logger.info(f"✅ Listed {len(blobs)} blobs in container {container_name}")
            return blobs
        
        except Exception as e:
            logger.error(f"❌ Failed to list blobs in container {container_name}: {e}")
            raise

    def delete(self, container_name: str, blob_path: str) -> bool:
        """
        Delete a blob from storage.
        
        Args:
            container_name: Name of the container
            blob_path: Path within the container
            
        Returns:
            True if deleted successfully, False if blob doesn't exist
        """
        try:
            self.get_blob_client(container_name, blob_path).delete_blob()
            logger.info(f"✅ Successfully deleted blob: {blob_path}")
            return True
        
        except ResourceNotFoundError:
            logger.warning(f"⚠️ Blob not found: {blob_path}")
            return False
        except Exception as e:
            logger.error(f"❌ Delete failed: {e}")
            raise

    def exists(self, container_name: str, blob_path: str) -> bool:
        """
        Check if a blob exists.
        
        Args:
            container_name: Name of the container
            blob_path: Path within the container
            
        Returns:
            True if blob exists, False otherwise
        """
        try:
            self.get_blob_client(container_name, blob_path).get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception as e:
            logger.error(f"❌ Error checking blob existence {blob_path}: {e}")
            raise

    def upload_image(self, container_name: str, folder: str, image_data: Union[bytes, BinaryIO], 
                     filename: Optional[str] = None, convert_to_jpeg: bool = False, jpeg_quality: int = 85) -> str:
        """
        Upload an image with optional conversion to JPEG.
        
        Args:
            container_name: Name of the container
            folder: Folder path within the container
            image_data: Image data (bytes or file-like object)
            filename: Optional filename (generates UUID if not provided)
            convert_to_jpeg: Whether to convert image to JPEG format
            jpeg_quality: JPEG quality (1-100) if converting
            
        Returns:
            URL of the uploaded image
        """
        try:
            if not filename:
                ext = ".jpg" if convert_to_jpeg else ".png"
                filename = f"{uuid.uuid4().hex}{ext}"

            if convert_to_jpeg:
                # Convert image to JPEG
                image = Image.open(io.BytesIO(image_data) if isinstance(image_data, bytes) else image_data).convert("RGB")

                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=jpeg_quality)
                buffer.seek(0)
                image_data = buffer
                content_type = "image/jpeg"
            else:
                content_type = "image/png"

            return self.put(container_name, f"{folder}/{filename}", image_data, content_type)
        
        except Exception as e:
            logger.error(f"❌ Upload image failed: {e}")
            raise

    def download_and_convert_image(self, src_container: str, src_path: str, dst_container: str, dst_folder: str,
                                   convert_to_jpeg: bool = True, jpeg_quality: int = 85) -> str:
        """
        Download an image from one container, convert it, and upload to another.
        
        Args:
            source_container: Source container name
            source_path: Source blob path
            dest_container: Destination container name
            dest_folder: Destination folder path
            convert_to_jpeg: Whether to convert to JPEG
            jpeg_quality: JPEG quality if converting
            
        Returns:
            URL of the converted and uploaded image
        """
        try:
            data = self.get(src_container, src_path)

            # Generate new filename
            new_filename = f"{uuid.uuid4().hex}.jpg" if convert_to_jpeg else f"{uuid.uuid4().hex}.png"
            dest_path = f"{dst_folder}/{new_filename}"

            return self.upload_image(
                dst_container,
                dst_folder,
                data,
                new_filename,
                convert_to_jpeg,
                jpeg_quality
            )
        
        except Exception as e:
            logger.error(f"❌ Failed to download and convert image from {src_path}: {e}")
            raise

    def copy(self, src_container: str, src_path: str, dst_container: str, dst_path: str) -> str:
        """
        Copy a blob from one location to another.
        
        Args:
            src_container: Source container name
            src_path: Source blob path
            dst_container: Destination container name
            dst_path: Destination blob path
            
        Returns:
            URL of the copied blob
        """
        try:
            source_url = self.generate_blob_url(src_container, src_path)
            self.get_blob_client(dst_container, dst_path).start_copy_from_url(source_url)
            logger.info(f"✅ Successfully copied blob from {src_path} to {dst_path}")
            return self.generate_blob_url(dst_container, dst_path)
        except Exception as e:
            logger.error(f"❌ Copy failed: {e}")
            raise

    def create_container(self, container_name: str) -> bool:
        """
        Create a new container.
        
        Args:
            container_name: Name of the container to create
            
        Returns:
            True if created successfully, False if already exists
        """
        try:
            self.get_container_client(container_name).create_container()
            logger.info(f"✅ Successfully created container: {container_name}")
            return True
        except ResourceExistsError:
            logger.info(f"ℹ️  Container already exists: {container_name}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to create container {container_name}: {e}")
            raise

    def delete_container(self, container_name: str) -> bool:
        """
        Delete a container and all its blobs.
        
        Args:
            container_name: Name of the container to delete
            
        Returns:
            True if deleted successfully, False if container doesn't exist
        """
        try:
            self.get_container_client(container_name).delete_container()
            logger.info(f"✅ Successfully deleted container: {container_name}")
            return True
        except ResourceNotFoundError:
            logger.warning(f"⚠️  Container not found for deletion: {container_name}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to delete container {container_name}: {e}")
            raise

# Global instance for easy access
_blob_storage_service = None

def get_blob_storage_service() -> BlobStorageService:
    """Get the global blob storage service instance."""
    global _blob_storage_service
    if _blob_storage_service is None:
        _blob_storage_service = BlobStorageService()
    return _blob_storage_service
