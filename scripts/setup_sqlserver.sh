#!/bin/bash
# ============================================================
# Setup SQL Server 2022 using Docker
# Updated to use new container name: face_check_sqlserver
# ============================================================

set -e

CONTAINER_NAME="face_check_sqlserver"
OLD_CONTAINER_NAME="sqlserver2022"
SA_PASSWORD="YourStrong@Passw0rd"
DB_NAME="FaceCheckDB"

echo "========================================"
echo "SQL Server 2022 Setup Script"
echo "========================================"

# Remove old container if exists
if docker ps -a --format '{{.Names}}' | grep -q "^${OLD_CONTAINER_NAME}$"; then
    echo "🗑️  Removing old container: $OLD_CONTAINER_NAME"
    docker stop $OLD_CONTAINER_NAME 2>/dev/null || true
    docker rm $OLD_CONTAINER_NAME 2>/dev/null || true
fi

# Check if new container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "ℹ️  Container $CONTAINER_NAME already exists"
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "✅ Container is already running"
    else
        echo "🚀 Starting existing container..."
        docker start $CONTAINER_NAME
    fi
else
    echo "🚀 Creating new SQL Server container..."
    docker run -d \
        --name $CONTAINER_NAME \
        -e "ACCEPT_EULA=Y" \
        -e "MSSQL_SA_PASSWORD=$SA_PASSWORD" \
        -e "MSSQL_PID=Express" \
        -p 1433:1433 \
        mcr.microsoft.com/mssql/server:2022-latest
    
    echo "⏳ Waiting for SQL Server to start (30s)..."
    sleep 30
fi

# Create database using mssql-tools container
echo "📦 Creating database $DB_NAME..."
docker run --rm \
    --network host \
    mcr.microsoft.com/mssql-tools:latest \
    /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$SA_PASSWORD" -Q "
        IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = '$DB_NAME')
        BEGIN
            CREATE DATABASE $DB_NAME;
            PRINT 'Database $DB_NAME created';
        END
        ELSE
            PRINT 'Database $DB_NAME already exists';
    "

# Create faces table
echo "📋 Creating faces table..."
docker run --rm \
    --network host \
    mcr.microsoft.com/mssql-tools:latest \
    /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$SA_PASSWORD" -d $DB_NAME -Q "
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'faces')
        BEGIN
            CREATE TABLE faces (
                id INT IDENTITY(1,1) PRIMARY KEY,
                user_id NVARCHAR(50) UNIQUE NOT NULL,
                name_user NVARCHAR(100),
                image VARBINARY(MAX),
                embedding VARBINARY(MAX),
                created_at DATETIME2 DEFAULT GETDATE()
            );
            PRINT 'Table faces created';
        END
        ELSE
            PRINT 'Table faces already exists';
    "

echo ""
echo "========================================"
echo "✅ SQL Server Setup Complete!"
echo "========================================"
echo "Container: $CONTAINER_NAME"
echo "Port: 1433"
echo "Database: $DB_NAME"
echo "User: sa"
echo "Password: $SA_PASSWORD"
echo ""
echo "Query command:"
echo "  docker exec $CONTAINER_NAME /opt/mssql-tools/bin/sqlcmd \\"
echo "    -S localhost -U sa -P \"$SA_PASSWORD\" \\"
echo "    -Q \"USE $DB_NAME; SELECT * FROM faces\""
echo "========================================"
