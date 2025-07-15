#!/usr/bin/env python3
"""
Quick test script for Classroom Management APIs

Simple script to test basic classroom operations.
Run this after starting the server: python main.py
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
ROBOT_ID = "robot_001"

def test_add_classroom_form(classroom_name, robot_id, device_id):
    """Test adding classroom with form data"""
    print(f"📝 Adding classroom '{classroom_name}' with robot '{robot_id}' and device '{device_id}' (form)...")
    
    try:
        data = {
            "classroom_name": classroom_name,
            "robot_id": robot_id,
            "device_id": device_id
        }
        response = requests.post(f"{BASE_URL}/classrooms/add/form", data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Added classroom: {result.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Failed to add classroom: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error adding classroom: {e}")
        return False

def test_add_classroom_json(classroom_name, robot_id, device_id):
    """Test adding classroom with JSON"""
    print(f"📝 Adding classroom '{classroom_name}' with robot '{robot_id}' and device '{device_id}' (JSON)...")
    
    try:
        json_data = {
            "classroom_name": classroom_name,
            "robot_id": robot_id,
            "device_id": device_id
        }
        response = requests.post(f"{BASE_URL}/classrooms/add", json=json_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Added classroom: {result.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Failed to add classroom: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error adding classroom: {e}")
        return False

def test_get_classrooms():
    """Test getting all classrooms"""
    print(f"📋 Getting all classrooms...")
    
    try:
        response = requests.get(f"{BASE_URL}/classrooms/list")
        
        if response.status_code == 200:
            result = response.json()
            classrooms = result.get('classrooms', [])
            print(f"✅ Found {len(classrooms)} classrooms:")
            for classroom in classrooms:
                print(f"   - Classroom: {classroom.get('classroom_name')}, Robot: {classroom.get('robot_id')}, Device: {classroom.get('device_id')}")
            return True
        else:
            print(f"❌ Failed to get classrooms: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error getting classrooms: {e}")
        return False

def test_get_classrooms_by_robot(robot_id):
    """Test getting classrooms for a specific robot"""
    print(f"📋 Getting classrooms for robot '{robot_id}'...")
    
    try:
        params = {"robot_id": robot_id}
        response = requests.get(f"{BASE_URL}/classrooms/list/by-robot", params=params)
        
        if response.status_code == 200:
            result = response.json()
            classrooms = result.get('classrooms', [])
            print(f"✅ Found {len(classrooms)} classrooms for robot '{robot_id}':")
            for classroom in classrooms:
                print(f"   - Classroom: {classroom.get('classroom_name')}, Device: {classroom.get('device_id')}")
            return True
        else:
            print(f"❌ Failed to get classrooms: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error getting classrooms: {e}")
        return False

def test_get_all_robots():
    """Test getting all robots (legacy endpoint)"""
    print(f"📋 Getting all robots...")
    
    try:
        response = requests.get(f"{BASE_URL}/classrooms/all")
        
        if response.status_code == 200:
            result = response.json()
            robots = result.get('robots', [])
            print(f"✅ Found {len(robots)} robots:")
            for robot in robots:
                robot_id = robot.get('robot_id', 'Unknown')
                devices = robot.get('device', [])
                print(f"   - Robot: {robot_id} ({len(devices)} classrooms)")
                for device in devices:
                    print(f"     * Device: {device.get('device_id')}, Classroom: {device.get('room_name')}")
            return True
        else:
            print(f"❌ Failed to get robots: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error getting robots: {e}")
        return False

def test_update_classroom(classroom_name, robot_id, device_id):
    """Test updating classroom"""
    print(f"✏️ Updating classroom '{classroom_name}' with robot '{robot_id}' and device '{device_id}'...")
    
    try:
        data = {
            "classroom_name": classroom_name,
            "robot_id": robot_id,
            "device_id": device_id
        }
        response = requests.put(f"{BASE_URL}/classrooms/update/form", data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Updated classroom: {result.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Failed to update classroom: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error updating classroom: {e}")
        return False

def test_delete_classroom(classroom_name):
    """Test deleting classroom"""
    print(f"🗑️ Deleting classroom '{classroom_name}'...")
    
    try:
        params = {"classroom_name": classroom_name}
        response = requests.delete(f"{BASE_URL}/classrooms/delete", params=params)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Deleted classroom: {result.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Failed to delete classroom: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error deleting classroom: {e}")
        return False

def test_validation_errors():
    """Test validation error handling"""
    print("🔍 Testing validation errors...")
    
    # Test empty robot_id
    print("  Testing empty robot_id...")
    data = {"classroom_name": "test", "robot_id": "", "device_id": "test"}
    response = requests.post(f"{BASE_URL}/classrooms/add", json=data)
    print(f"    Status: {response.status_code} - Expected: 400")
    
    # Test empty device_id
    print("  Testing empty device_id...")
    data = {"classroom_name": "test", "robot_id": "test", "device_id": ""}
    response = requests.post(f"{BASE_URL}/classrooms/add", json=data)
    print(f"    Status: {response.status_code} - Expected: 400")
    
    # Test empty classroom_name
    print("  Testing empty classroom_name...")
    data = {"classroom_name": "", "robot_id": "test", "device_id": "test"}
    response = requests.post(f"{BASE_URL}/classrooms/add", json=data)
    print(f"    Status: {response.status_code} - Expected: 400")

def main():
    print("🚀 Starting Quick Classroom API Tests")
    print("=" * 50)
    
    # Test data
    test_classrooms = [
        ("Classroom A", "robot_001", "tv_001"),
        ("Classroom B", "robot_002", "tv_002"),
        ("Classroom C", "robot_003", "tv_003")
    ]
    
    # Test adding classrooms
    print("\n📝 Testing Add Classroom Operations:")
    for classroom_name, robot_id, device_id in test_classrooms:
        test_add_classroom_form(classroom_name, robot_id, device_id)
        test_add_classroom_json(classroom_name, robot_id, device_id)
    
    # Test getting classrooms
    print("\n📋 Testing Get Classroom Operations:")
    test_get_classrooms()
    test_get_classrooms_by_robot("robot_001")
    
    # Test updating classrooms
    print("\n✏️ Testing Update Classroom Operations:")
    test_update_classroom("Classroom A", "robot_001", "tv_updated_001")
    
    # Test validation
    print("\n🔍 Testing Validation:")
    test_validation_errors()
    
    # Test deleting classrooms
    print("\n🗑️ Testing Delete Classroom Operations:")
    for classroom_name, robot_id, device_id in test_classrooms:
        test_delete_classroom(classroom_name)
    
    print("\n✅ Quick tests completed!")

if __name__ == "__main__":
    main() 