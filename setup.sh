#!/bin/bash

# Sales Follow-Up Assistant - Quick Setup Script

echo "🚀 Setting up Sales Follow-Up Assistant..."

# Create project directory structure
echo "📁 Creating directory structure..."
mkdir -p app/agent
mkdir -p app/services
mkdir -p app/utils
mkdir -p data

# Create empty __init__.py files
touch app/__init__.py
touch app/agent/__init__.py
touch app/services/__init__.py
touch app/utils/__init__.py

# Create sample data files
echo "📊 Creating sample data files..."

cat > data/orders.csv << 'EOF'
customer_id,order_id,order_date,sku,qty,price
C001,SO-101,2025-08-20,CAKE-CHOC,3,12.50
C001,SO-122,2025-09-05,COOK-OAT,5,2.10
C002,SO-130,2025-09-01,JUICE-ORG,10,1.20
C003,SO-140,2025-07-30,CAKE-CHOC,1,12.50
C003,SO-155,2025-09-10,COFF-BEAN,2,7.90
C001,SO-160,2025-09-12,CAKE-CHOC,1,12.50
C004,SO-170,2025-08-01,TEA-GREEN,4,3.50
EOF

cat > data/customers.csv << 'EOF'
customer_id,name,segment,territory,credit_terms
C001,Gourmet Gateway,HO.RE.CA,West,NET15
C002,Snack Shack,Retail,East,PREPAID
C003,Daily Delights,Retail,North,NET30
C004,Leaf & Cup,Cafe,South,NET15
EOF

# Create .env file from example
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.example .env
    echo "🔑 Please update .env with your AWS credentials!"
else
    echo "✅ .env file already exists"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo "📦 Installing dependencies..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || {
    echo "⚠️  Please activate virtual environment manually:"
    echo "   Linux/Mac: source venv/bin/activate"
    echo "   Windows: venv\\Scripts\\activate"
    echo "   Then run: pip install -r requirements.txt"
    exit 1
}

pip install -r requirements.txt

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Update .env with your AWS credentials"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the application: python run.py"
echo "4. Open http://localhost:8000/docs for API documentation"
echo ""
echo "🔗 Test endpoints:"
echo "   POST /analyze - Analyze customer"
echo "   POST /top-followups - Get daily follow-ups"
echo "   GET /health - Health check"
echo ""