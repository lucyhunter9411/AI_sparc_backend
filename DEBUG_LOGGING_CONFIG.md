# Debug Logging Configuration

This document explains how to configure debug logging for the lesson workflow system.

## Environment Variable

The debug logging is controlled by the `DEBUG_LESSON_WORKFLOW` environment variable.

### Values
- **Enable**: `true`, `1`, `yes` (case-insensitive)
- **Disable**: `false`, `0`, `no` (case-insensitive), or omit the variable

### Examples
```bash
# Enable debug logging
DEBUG_LESSON_WORKFLOW=true
DEBUG_LESSON_WORKFLOW=1
DEBUG_LESSON_WORKFLOW=yes

# Disable debug logging
DEBUG_LESSON_WORKFLOW=false
DEBUG_LESSON_WORKFLOW=0
DEBUG_LESSON_WORKFLOW=no
# Or simply don't set the variable
```

## Local Development

### Option 1: .env file
Create a `.env` file in the project root with:
```bash
DEBUG_LESSON_WORKFLOW=true
```

### Option 2: Export in shell
```bash
export DEBUG_LESSON_WORKFLOW=true
python main.py
```

## Cloud Deployment (Azure App Service)

### Azure Portal Configuration
1. Go to your App Service in Azure Portal
2. Navigate to **Settings** → **Configuration**
3. Under **Application settings**, add:
   - **Name**: `DEBUG_LESSON_WORKFLOW`
   - **Value**: `true` (or `false` to disable)
4. Click **Save**

### Azure CLI Configuration
```bash
az webapp config appsettings set \
  --name your-app-name \
  --resource-group your-resource-group \
  --settings DEBUG_LESSON_WORKFLOW=true
```

### Environment Variable Priority
1. Azure App Service Configuration (highest priority)
2. .env file (if present)
3. Default value (false)

## What Gets Logged

When `DEBUG_LESSON_WORKFLOW=true`, the system logs:

### Audio Routing Decisions
- Which robot is being processed
- Number of connected audio clients
- Whether audio client is connected
- Which audio path is taken (audio client vs frontend)
- Content index and text preview

### WebSocket Message Flow
- Message direction (SEND/RECEIVE)
- Message type
- Target client
- Reason for routing decision

### State Machine Transitions
- State changes
- Transition triggers
- Robot context

## Performance Impact

- **When disabled**: Zero performance impact - no debug logging occurs
- **When enabled**: Minimal performance impact - only logs when debug is enabled
- **Production recommendation**: Keep disabled unless debugging specific issues

## Testing the Configuration

Run the test script to verify debug logging is working:
```bash
# Test with debug enabled
DEBUG_LESSON_WORKFLOW=true python test_debug_logging.py

# Test with debug disabled
DEBUG_LESSON_WORKFLOW=false python test_debug_logging.py
```

## Troubleshooting

### Debug logging not working?
1. Check if `DEBUG_LESSON_WORKFLOW` is set correctly
2. Verify the environment variable is loaded (check logs for config loading)
3. Ensure the app has been restarted after changing the setting

### Too many debug messages?
- Set `DEBUG_LESSON_WORKFLOW=false` to disable
- Or modify the logging level in your logging configuration
