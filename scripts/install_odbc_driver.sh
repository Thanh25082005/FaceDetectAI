#!/bin/bash
# =============================================================================
# ODBC Driver 18 Installation for Ubuntu 24.04
# Required for Python to connect to SQL Server
# =============================================================================

set -e

echo "🔧 Installing ODBC Driver 18 for SQL Server"
echo "============================================"

# Install prerequisites
echo "📦 Installing prerequisites..."
sudo apt-get update
sudo apt-get install -y curl apt-transport-https unixodbc-dev

# Add Microsoft repository (using Ubuntu 22.04 packages as workaround for 24.04)
echo "📥 Adding Microsoft repository..."
curl https://packages.microsoft.com/keys/microsoft.asc | sudo tee /etc/apt/trusted.gpg.d/microsoft.asc

# For Ubuntu 24.04, use Ubuntu 22.04 packages (workaround)
echo "deb [arch=amd64] https://packages.microsoft.com/ubuntu/22.04/prod jammy main" | \
    sudo tee /etc/apt/sources.list.d/mssql-release.list

# Install ODBC Driver
echo "📦 Installing ODBC Driver 18..."
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Install optional tools
echo "📦 Installing SQL Server tools..."
sudo ACCEPT_EULA=Y apt-get install -y mssql-tools18

# Add tools to PATH
echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc

echo ""
echo "✅ ODBC Driver 18 installed successfully!"
echo ""
echo "Verify installation:"
echo "   odbcinst -q -d -n 'ODBC Driver 18 for SQL Server'"
echo ""
echo "Test connection:"
echo "   python -c \"from database.connection import test_connection; import asyncio; asyncio.run(test_connection())\""
