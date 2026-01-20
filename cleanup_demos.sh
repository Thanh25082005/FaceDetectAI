#!/bin/bash
# Cleanup Project - Giữ backend + demo_pipeline.py
# Backup trước khi xóa

echo "=================================================="
echo "🧹 PROJECT CLEANUP"
echo "=================================================="
echo ""

# Create backup directory
BACKUP_DIR="backup_demos_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup in: $BACKUP_DIR"
echo ""

# Files to DELETE (backup first)
FILES_TO_DELETE=(
    # Old demos
    "demo_streamlit.py"
    "demo_modern.py"
    "demo_realtime.py"
    "demo_optimized_pipeline.py"
    "demo_auto_register.py"
    "demo_headless_auto.py"
    "demo_headless.py"
    "demo_webcam.py"
    "demo_client.py"
    
    # Old guides (không liên quan pipeline)
    "DEMO_MODERN_GUIDE.md"
    "REALTIME_SCANNER_GUIDE.md"
    "README_AUTO_DEMO.md"
    "OPTIMIZED_PIPELINE_GUIDE.md"
    "RUNNING_SERVICES.md"
    
    # Test scripts không cần
    "test_demo.sh"
    "test_system.py"
)

# Files to KEEP
echo "✅ FILES TO KEEP:"
echo "   Backend:"
echo "   - main.py"
echo "   - config.py"
echo "   - api/"
echo "   - models/"
echo "   - streaming/"
echo "   - utils/"
echo "   - data/"
echo ""
echo "   Demo:"
echo "   - demo_pipeline.py"
echo ""
echo "   Utilities:"
echo "   - quick_add_face.py"
echo "   - test_auto_stop_logic.py"
echo "   - reset_database.sh"
echo ""
echo "   Docs:"
echo "   - README.md"
echo "   - TEST_AUTO_STOP.md"
echo "   - TROUBLESHOOTING_RECOGNITION.md"
echo "   - PROCESS_FRAME_FLOW.md"
echo ""
echo "=================================================="
echo ""

# Backup and delete
echo "🗑️  FILES TO DELETE (will backup first):"
for file in "${FILES_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "   - $file"
        cp "$file" "$BACKUP_DIR/"
        rm "$file"
    fi
done

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Summary:"
echo "   - Backup location: $BACKUP_DIR"
echo "   - Deleted: ${#FILES_TO_DELETE[@]} demo files"
echo "   - Kept: Backend + demo_pipeline.py + utilities"
echo ""
echo "🎯 Your project now contains:"
echo "   - Full backend (API + models)"
echo "   - demo_pipeline.py (main demo)"
echo "   - Essential utilities"
echo ""
