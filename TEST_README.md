# Classroom API Testing Guide

This guide explains how to test the Classroom Management APIs using the provided test scripts.

## Prerequisites

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Server**
   ```bash
   python main.py
   ```
   The server should start on `http://localhost:8000`

## Test Scripts

### 1. Quick Test Script (`quick_test_classrooms.py`)

A simple, synchronous test script for basic functionality testing.

**Usage:**
```bash
python quick_test_classrooms.py
```

**Features:**
- ✅ Health check
- ✅ Add classrooms (form data)
- ✅ Add classrooms (JSON)
- ✅ Get classrooms for a robot
- ✅ Get all robots
- ✅ Update classrooms
- ✅ Delete classrooms
- ✅ Invalid input testing

**Output Example:**
```
🚀 Starting Quick Classroom API Tests
📡 Base URL: http://localhost:8000
🤖 Robot ID: test_robot_001
==================================================
🔍 Testing health endpoint...
✅ Health check passed

📝 Testing Add Operations
📝 Adding classroom 'Classroom A' with device 'device_001' (form)...
✅ Added classroom: New robot_id created and device added
...
```

### 2. Comprehensive Test Script (`test_classroom_apis.py`)

A comprehensive, asynchronous test script with detailed reporting and error handling.

**Usage:**
```bash
# Basic usage
python test_classroom_apis.py

# Custom server URL
python test_classroom_apis.py --base-url http://localhost:8000

# Custom robot ID
python test_classroom_apis.py --robot-id my_robot_001

# Both custom parameters
python test_classroom_apis.py --base-url http://localhost:8000 --robot-id my_robot_001
```

**Features:**
- ✅ All quick test features
- ✅ Async HTTP client for better performance
- ✅ Detailed test reporting with timestamps
- ✅ Test summary with success rates
- ✅ Better error handling and logging
- ✅ Command-line argument support
- ✅ Graceful interruption handling

**Output Example:**
```
🚀 Starting Classroom API Tests
📡 Base URL: http://localhost:8000
🤖 Robot ID: test_robot_001
============================================================
[14:30:15] ✅ PASS - Health Check

📝 Testing Add Classroom Operations
[14:30:15] ✅ PASS - Add Classroom (Form) - classroom_101
    Details: Message: New robot_id created and device added

[14:30:16] ✅ PASS - Add Classroom (JSON) - classroom_101
    Details: Message: Device added to existing robot_id

...

============================================================
TEST SUMMARY
============================================================
Total Tests: 25
Passed: 25
Failed: 0
Success Rate: 100.0%
============================================================
```

## API Endpoints Tested

### Form Data Endpoints
- `POST /classrooms/add_or_update/` - Add or update classroom
- `DELETE /classrooms/delete/` - Delete classroom
- `GET /classrooms/get/` - Get classrooms for a robot

### JSON Endpoints
- `POST /classrooms/add_or_update/json/` - Add or update classroom (JSON)
- `DELETE /classrooms/delete/json/` - Delete classroom (JSON)

### Other Endpoints
- `GET /classrooms/all/` - Get all robots with classrooms
- `GET /health` - Health check

## Test Scenarios

### 1. **Basic CRUD Operations**
- ✅ Create new classrooms
- ✅ Read classroom lists
- ✅ Update existing classrooms
- ✅ Delete classrooms

### 2. **Data Validation**
- ✅ Empty robot_id (should return 400)
- ✅ Empty device_id (should return 400)
- ✅ Empty classroom_name (should return 400)

### 3. **Error Handling**
- ✅ Non-existent robot (should return 404)
- ✅ Server connectivity issues
- ✅ Invalid JSON payloads

### 4. **Edge Cases**
- ✅ Adding duplicate device_id (should update)
- ✅ Deleting non-existent classroom
- ✅ Multiple robots with multiple classrooms

## Configuration

### Environment Variables
The test scripts use these default values:
- `BASE_URL`: `http://localhost:8000`
- `ROBOT_ID`: `test_robot_001`

### Custom Test Data
You can modify the test data in the scripts:

**Quick Test:**
```python
test_classrooms = [
    ("Classroom A", "device_001"),
    ("Classroom B", "device_002"),
    ("Classroom C", "device_003")
]
```

**Comprehensive Test:**
```python
test_classrooms = [
    ("classroom_101", "device_101"),
    ("classroom_102", "device_102"),
    ("classroom_103", "device_103")
]
```

## Troubleshooting

### Common Issues

1. **Server Not Running**
   ```
   ❌ Health check failed. Make sure the server is running.
   ```
   **Solution:** Start the FastAPI server with `python main.py`

2. **Connection Refused**
   ```
   ❌ Health check error: Connection refused
   ```
   **Solution:** Check if the server is running on the correct port

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'httpx'
   ```
   **Solution:** Install dependencies with `pip install -r requirements.txt`

4. **Database Connection Issues**
   ```
   ❌ Failed to add classroom: 500 - Internal server error
   ```
   **Solution:** Check MongoDB connection and database configuration

### Debug Mode

For more detailed error information, you can modify the test scripts to include debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with CI/CD

The comprehensive test script can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Test Classroom APIs
  run: |
    python test_classroom_apis.py --base-url ${{ env.API_URL }}
```

## Performance Testing

For performance testing, you can modify the scripts to:
- Test with larger datasets
- Add concurrent request testing
- Measure response times
- Test under load

Example performance test addition:
```python
import time

start_time = time.time()
response = await session.post(url, json=data)
end_time = time.time()
print(f"Response time: {end_time - start_time:.3f}s")
```

## Contributing

When adding new test cases:
1. Follow the existing naming conventions
2. Add appropriate error handling
3. Include both positive and negative test cases
4. Update this README with new features
5. Ensure tests are idempotent (can be run multiple times) 