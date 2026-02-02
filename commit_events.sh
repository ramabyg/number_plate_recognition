#!/bin/bash
# Script to commit TensorFlow events to git for analysis
# Usage: ./commit_events.sh "Description of training run"

if [ -z "$1" ]; then
    echo "Usage: ./commit_events.sh \"Description of training run\""
    echo ""
    echo "Example:"
    echo "  ./commit_events.sh \"Training with learning_rate=0.0001\""
    exit 1
fi

DESCRIPTION="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "Committing TensorFlow events..."
echo "Description: $DESCRIPTION"
echo "Timestamp: $TIMESTAMP"

# Add TensorFlow events
git add tf_events/

# Add model metrics if available
if [ -f "output_pjt3_retinanet/metrics.json" ]; then
    git add output_pjt3_retinanet/metrics.json
fi

# Commit with descriptive message
git commit -m "TensorFlow Events: $DESCRIPTION

Timestamp: $TIMESTAMP
Location: tf_events/, output_pjt3_retinanet/metrics.json
"

if [ $? -eq 0 ]; then
    echo "✓ Events committed successfully!"
    echo "Run 'git log' to view the commit history."
else
    echo "No changes to commit or error occurred."
fi
