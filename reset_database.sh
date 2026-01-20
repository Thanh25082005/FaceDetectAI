#!/bin/bash
# Reset Database - Xóa tất cả faces và checkins

echo "=================================================="
echo "🗑️  RESET DATABASE"
echo "=================================================="
echo ""
echo "⚠️  WARNING: This will DELETE ALL:"
echo "   - Face embeddings (faces.db)"
echo "   - Check-in history (checkins.db)"
echo ""
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" = "yes" ]; then
    echo ""
    echo "🗑️  Deleting databases..."
    
    # Backup first
    if [ -f "data/faces.db" ]; then
        cp data/faces.db data/faces.db.backup.$(date +%Y%m%d_%H%M%S)
        echo "✅ Backed up faces.db"
    fi
    
    if [ -f "data/checkins.db" ]; then
        cp data/checkins.db data/checkins.db.backup.$(date +%Y%m%d_%H%M%S)
        echo "✅ Backed up checkins.db"
    fi
    
    # Delete
    rm -f data/faces.db
    rm -f data/checkins.db
    
    echo ""
    echo "✅ Database deleted!"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Restart API server (database will auto-recreate)"
    echo "   2. Add new faces using Enrollment mode"
    echo ""
else
    echo ""
    echo "❌ Cancelled. Database not deleted."
fi
