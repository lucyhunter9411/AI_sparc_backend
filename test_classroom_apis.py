#!/usr/bin/env python3
"""
Test script for Classroom Management APIs

This script tests all the classroom management endpoints including:
- Adding/updating classrooms
- Deleting classrooms  
- Getting classrooms for a robot
- Getting all classrooms

Usage:
    python test_classroom_apis.py [--base-url BASE_URL] [--robot-id ROBOT_ID]
"""

import asyncio
import httpx
import json
import sys
import argparse
from typing import Dict, Any, Optional
from datetime import datetime

class ClassroomAPITester:
    """Test class for Classroom Management APIs"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.test_results = []
        self.session: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    def log_test(self, test_name: str, success: bool, response: httpx.Response, details: str = ""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        result = {
            "timestamp": timestamp,
            "test": test_name,
            "status": status,
            "status_code": response.status_code,
            "success": success,
            "details": details
        }
        
        self.test_results.append(result)
        
        print(f"[{timestamp}] {status} - {test_name}")
        if not success:
            print(f"    Status Code: {response.status_code}")
            print(f"    Response: {response.text}")
            if details:
                print(f"    Details: {details}")
        print()
    
    async def test_health_check(self) -> bool:
        """Test the health check endpoint"""
        try:
            response = await self.session.get(f"{self.base_url}/health")
            success = response.status_code == 200
            self.log_test("Health Check", success, response)
            return success
        except Exception as e:
            self.log_test("Health Check", False, httpx.Response(500), f"Exception: {e}")
            return False
    
    async def test_add_classroom_form(self, robot_id: str, device_id: str, classroom_name: str) -> bool:
        """Test adding a classroom using form data"""
        try:
            data = {
                "classroom_name": classroom_name,
                "robot_id": robot_id,
                "device_id": device_id
            }
            response = await self.session.post(f"{self.base_url}/classrooms/add/form", data=data)
            
            success = response.status_code == 200
            details = ""
            if success:
                response_data = response.json()
                details = f"Message: {response_data.get('message', 'N/A')}"
            
            self.log_test(f"Add Classroom (Form) - {classroom_name}", success, response, details)
            return success
        except Exception as e:
            self.log_test(f"Add Classroom (Form) - {classroom_name}", False, httpx.Response(500), f"Exception: {e}")
            return False
    
    async def test_add_classroom_json(self, robot_id: str, device_id: str, classroom_name: str) -> bool:
        """Test adding a classroom using JSON body"""
        try:
            json_data = {
                "classroom_name": classroom_name,
                "robot_id": robot_id,
                "device_id": device_id
            }
            response = await self.session.post(f"{self.base_url}/classrooms/add", json=json_data)
            
            success = response.status_code == 200
            details = ""
            if success:
                response_data = response.json()
                details = f"Message: {response_data.get('message', 'N/A')}"
            
            self.log_test(f"Add Classroom (JSON) - {classroom_name}", success, response, details)
            return success
        except Exception as e:
            self.log_test(f"Add Classroom (JSON) - {classroom_name}", False, httpx.Response(500), f"Exception: {e}")
            return False
    
    async def test_update_classroom_form(self, robot_id: str, device_id: str, classroom_name: str) -> bool:
        """Test updating a classroom using form data"""
        try:
            data = {
                "classroom_name": classroom_name,
                "robot_id": robot_id,
                "device_id": device_id
            }
            response = await self.session.put(f"{self.base_url}/classrooms/update/form", data=data)
            
            success = response.status_code == 200
            details = ""
            if success:
                response_data = response.json()
                details = f"Message: {response_data.get('message', 'N/A')}"
            
            self.log_test(f"Update Classroom (Form) - {classroom_name}", success, response, details)
            return success
        except Exception as e:
            self.log_test(f"Update Classroom (Form) - {classroom_name}", False, httpx.Response(500), f"Exception: {e}")
            return False
    
    async def test_get_classrooms_by_robot(self, robot_id: str) -> bool:
        """Test getting classrooms for a specific robot"""
        try:
            params = {"robot_id": robot_id}
            response = await self.session.get(f"{self.base_url}/classrooms/list/by-robot", params=params)
            
            success = response.status_code == 200
            details = ""
            if success:
                response_data = response.json()
                classrooms = response_data.get('classrooms', [])
                details = f"Found {len(classrooms)} classrooms for robot {robot_id}"
            
            self.log_test(f"Get Classrooms by Robot - {robot_id}", success, response, details)
            return success
        except Exception as e:
            self.log_test(f"Get Classrooms by Robot - {robot_id}", False, httpx.Response(500), f"Exception: {e}")
            return False
    
    async def test_get_all_classrooms(self) -> bool:
        """Test getting all classrooms"""
        try:
            response = await self.session.get(f"{self.base_url}/classrooms/list")
            
            success = response.status_code == 200
            details = ""
            if success:
                response_data = response.json()
                classrooms = response_data.get('classrooms', [])
                details = f"Found {len(classrooms)} total classrooms"
            
            self.log_test("Get All Classrooms", success, response, details)
            return success
        except Exception as e:
            self.log_test("Get All Classrooms", False, httpx.Response(500), f"Exception: {e}")
            return False
    
    async def test_delete_classroom(self, classroom_name: str) -> bool:
        """Test deleting a classroom by name"""
        try:
            params = {"classroom_name": classroom_name}
            response = await self.session.delete(f"{self.base_url}/classrooms/delete", params=params)
            
            success = response.status_code == 200
            details = ""
            if success:
                response_data = response.json()
                details = f"Message: {response_data.get('message', 'N/A')}"
            
            self.log_test(f"Delete Classroom - {classroom_name}", success, response, details)
            return success
        except Exception as e:
            self.log_test(f"Delete Classroom - {classroom_name}", False, httpx.Response(500), f"Exception: {e}")
            return False
    
    async def test_invalid_inputs(self) -> bool:
        """Test various invalid input scenarios"""
        tests_passed = 0
        total_tests = 3
        
        # Test empty robot_id
        try:
            json_data = {"classroom_name": "test", "robot_id": "", "device_id": "test"}
            response = await self.session.post(f"{self.base_url}/classrooms/add", json=json_data)
            if response.status_code == 400:
                tests_passed += 1
                self.log_test("Invalid Input - Empty Robot ID", True, response, "Correctly rejected")
            else:
                self.log_test("Invalid Input - Empty Robot ID", False, response, f"Expected 400, got {response.status_code}")
        except Exception as e:
            self.log_test("Invalid Input - Empty Robot ID", False, httpx.Response(500), f"Exception: {e}")
        
        # Test empty device_id
        try:
            json_data = {"classroom_name": "test", "robot_id": "test", "device_id": ""}
            response = await self.session.post(f"{self.base_url}/classrooms/add", json=json_data)
            if response.status_code == 400:
                tests_passed += 1
                self.log_test("Invalid Input - Empty Device ID", True, response, "Correctly rejected")
            else:
                self.log_test("Invalid Input - Empty Device ID", False, response, f"Expected 400, got {response.status_code}")
        except Exception as e:
            self.log_test("Invalid Input - Empty Device ID", False, httpx.Response(500), f"Exception: {e}")
        
        # Test empty classroom_name
        try:
            json_data = {"classroom_name": "", "robot_id": "test", "device_id": "test"}
            response = await self.session.post(f"{self.base_url}/classrooms/add", json=json_data)
            if response.status_code == 400:
                tests_passed += 1
                self.log_test("Invalid Input - Empty Classroom Name", True, response, "Correctly rejected")
            else:
                self.log_test("Invalid Input - Empty Classroom Name", False, response, f"Expected 400, got {response.status_code}")
        except Exception as e:
            self.log_test("Invalid Input - Empty Classroom Name", False, httpx.Response(500), f"Exception: {e}")
        
        return tests_passed == total_tests
    
    async def test_nonexistent_robot(self) -> bool:
        """Test getting classrooms for a non-existent robot"""
        try:
            params = {"robot_id": "nonexistent_robot_999"}
            response = await self.session.get(f"{self.base_url}/classrooms/list/by-robot", params=params)
            
            success = response.status_code == 200
            details = ""
            if success:
                response_data = response.json()
                classrooms = response_data.get('classrooms', [])
                details = f"Found {len(classrooms)} classrooms (expected 0)"
            
            self.log_test("Non-existent Robot", success, response, details)
            return success
        except Exception as e:
            self.log_test("Non-existent Robot", False, httpx.Response(500), f"Exception: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if failed_tests > 0:
            print("\nFailed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']} (Status: {result['status_code']})")
        
        print("=" * 60)

async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Classroom Management APIs")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for the API")
    parser.add_argument("--robot-id", default="test_robot_001", help="Robot ID for testing")
    
    args = parser.parse_args()
    
    print("🚀 Starting Classroom API Tests")
    print(f"📡 Base URL: {args.base_url}")
    print(f"🤖 Robot ID: {args.robot_id}")
    print("=" * 60)
    
    async with ClassroomAPITester(args.base_url) as tester:
        # Test health check first
        if not await tester.test_health_check():
            print("❌ Health check failed. Make sure the server is running.")
            return
        
        # Test data
        test_classrooms = [
            ("classroom_101", "robot_001", "tv_001"),
            ("classroom_102", "robot_002", "tv_002"),
            ("classroom_103", "robot_003", "tv_003")
        ]
        
        # Test adding classrooms
        print("📝 Testing Add Classroom Operations")
        for classroom_name, robot_id, device_id in test_classrooms:
            await tester.test_add_classroom_form(robot_id, device_id, classroom_name)
            await tester.test_add_classroom_json(robot_id, device_id, classroom_name)
        
        # Test getting classrooms
        print("📋 Testing Get Classroom Operations")
        await tester.test_get_classrooms_by_robot(args.robot_id)
        await tester.test_get_all_classrooms()
        
        # Test updating classrooms
        print("✏️ Testing Update Classroom Operations")
        await tester.test_update_classroom_form("robot_001", "tv_updated_001", "Updated Classroom 101")
        
        # Test invalid inputs
        print("⚠️ Testing Invalid Inputs")
        await tester.test_invalid_inputs()
        
        # Test non-existent robot
        print("🔍 Testing Non-existent Robot")
        await tester.test_nonexistent_robot()
        
        # Test deleting classrooms
        print("🗑️ Testing Delete Classroom Operations")
        for classroom_name, robot_id, device_id in test_classrooms:
            await tester.test_delete_classroom(classroom_name)
        
        # Print summary
        tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main()) 