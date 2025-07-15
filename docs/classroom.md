# RAG Backend Lucy - Classroom Management System

A FastAPI-based backend system for managing classrooms, robots, and devices (TVs) in an educational environment. The system provides a RESTful API for classroom management with real-time device communication capabilities.

## 🏗️ Architecture

The system follows a classroom-centric data model where:
- **Each classroom** has exactly **one robot** and **one device (TV)**
- **Classrooms** are the primary entities (not robots or devices)
- **Device IDs** represent TVs that can display content
- **Robot IDs** represent the robots that control each classroom

## 📊 Data Model

### Classroom Structure
```json
{
  "classroom_name": "Room 101",
  "robot_id": "robot_001", 
  "device_id": "tv_001"
}
```

### Database Collections
- **`classrooms`** - Stores classroom information
- **`devices`** - Legacy collection (deprecated)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- MongoDB
- FastAPI
- Required Python packages (see `requirements.txt`)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd RAG_Backend_lucy

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the server
python main.py
```

The server will start on `http://localhost:8000`

## 📚 API Documentation

### Base URL
```
http://localhost:8000/classrooms
```

### Authentication
Most endpoints require no authentication. The system uses environment variables for TV app authentication.

## 🔌 API Endpoints

### 1. Add Classroom
- **Method:** POST
- **Path:** `/classrooms/add`
- **Description:** Create a new classroom with a robot and device (TV).
- **Request Body (JSON):**
  - `classroom_name` (string, required)
  - `robot_id` (string, required)
  - `device_id` (string, required)
- **Example:**
```json
{
  "classroom_name": "Room 101",
  "robot_id": "robot_001",
  "device_id": "tv_001"
}
```
- **Response:**
```json
{
  "message": "Classroom created successfully",
  "success": true
}
```

---

### 2. Update Classroom
- **Method:** PUT
- **Path:** `/classrooms/update`
- **Description:** Update the robot or device for an existing classroom.
- **Request Body (JSON):**
  - `classroom_name` (string, required)
  - `robot_id` (string, required)
  - `device_id` (string, required)
- **Example:**
```json
{
  "classroom_name": "Room 101",
  "robot_id": "robot_002",
  "device_id": "tv_002"
}
```
- **Response:**
```json
{
  "message": "Classroom updated successfully",
  "success": true
}
```

---

### 3. Get All Classrooms
- **Method:** GET
- **Path:** `/classrooms/list`
- **Description:** Retrieve a list of all classrooms.
- **Response:**
```json
{
  "classrooms": [
    {
      "classroom_name": "Room 101",
      "robot_id": "robot_001",
      "device_id": "tv_001"
    },
    {
      "classroom_name": "Room 102",
      "robot_id": "robot_002",
      "device_id": "tv_002"
    }
  ]
}
```

---

### 4. Get Classrooms by Robot
- **Method:** GET
- **Path:** `/classrooms/list/by-robot`
- **Description:** Retrieve all classrooms associated with a specific robot.
- **Query Parameter:**
  - `robot_id` (string, required)
- **Example:**
`/classrooms/list/by-robot?robot_id=robot_001`
- **Response:**
```json
{
  "classrooms": [
    {
      "classroom_name": "Room 101",
      "robot_id": "robot_001",
      "device_id": "tv_001"
    }
  ]
}
```

---

### 5. Get Specific Classroom
- **Method:** GET
- **Path:** `/classrooms/get`
- **Description:** Retrieve details for a specific classroom.
- **Query Parameter:**
  - `classroom_name` (string, required)
- **Example:**
`/classrooms/get?classroom_name=Room 101`
- **Response:**
```json
{
  "classroom": {
    "classroom_name": "Room 101",
    "robot_id": "robot_001",
    "device_id": "tv_001"
  }
}
```

---

### 6. Delete Classroom
- **Method:** DELETE
- **Path:** `/classrooms/delete`
- **Description:** Delete a classroom by name.
- **Query Parameter:**
  - `classroom_name` (string, required)
- **Example:**
`/classrooms/delete?classroom_name=Room 101`
- **Response:**
```json
{
  "message": "Classroom deleted successfully",
  "success": true
}
```

## 🎯 React TypeScript Client API Usage Examples

Below are concise examples of how to call the classroom management APIs from a React TypeScript client using axios. These snippets assume you have axios installed and imported.

### 1. Add a Classroom
```typescript
import axios from 'axios';

await axios.post('http://localhost:8000/classrooms/add', {
  classroom_name: 'Room 101',
  robot_id: 'robot_001',
  device_id: 'tv_001',
});
```

### 2. Update a Classroom
```typescript
await axios.put('http://localhost:8000/classrooms/update', {
  classroom_name: 'Room 101',
  robot_id: 'robot_002',
  device_id: 'tv_002',
});
```

### 3. Get All Classrooms
```typescript
const response = await axios.get('http://localhost:8000/classrooms/list');
console.log(response.data.classrooms);
```

### 4. Get Classrooms by Robot
```typescript
const response = await axios.get('http://localhost:8000/classrooms/list/by-robot', {
  params: { robot_id: 'robot_001' },
});
console.log(response.data.classrooms);
```

### 5. Get a Specific Classroom
```typescript
const response = await axios.get('http://localhost:8000/classrooms/get', {
  params: { classroom_name: 'Room 101' },
});
console.log(response.data.classroom);
```

### 6. Delete a Classroom
```typescript
await axios.delete('http://localhost:8000/classrooms/delete', {
  params: { classroom_name: 'Room 101' },
});
```

---

For more details on request/response formats, see the API documentation above.

## 🧪 Testing

### Run Backend Tests
```bash
# Quick tests
python quick_test_classrooms.py

# Comprehensive tests
python test_classroom_apis.py
```

### Run Frontend Tests
```bash
cd classroom-client
npm test
```

## 🔧 Environment Variables

Create a `.env` file with the following variables:

```env
# MongoDB Configuration
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=rag_backend

# TV App Configuration
TV_APP_API=http://your-tv-app-api.com
TV_APP_WEBSOCKET=ws://your-tv-app-websocket.com
TV_APP_LOGIN_EMAIL=your-email@example.com
TV_APP_LOGIN_PASSWORD=your-password

# Base URL for image serving
BASE_URL=http://localhost:8000
```

## 📝 API Response Codes

- **200** - Success
- **400** - Bad Request (validation errors)
- **404** - Not Found
- **500** - Internal Server Error

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.