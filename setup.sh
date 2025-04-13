#!/bin/bash

# Resume Optimizer Setup Script

echo "====== Setting up Resume Optimizer ======"
echo "This script will install all necessary dependencies for both the backend and frontend."

# Create and activate Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install openai flask flask-cors werkzeug uuid nltk scikit-learn pandas numpy pypdf docx2txt

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd resume-optimizer
npm install

echo "====== Setup complete! ======"
echo ""
echo "To run the application:"
echo "1. Start the Flask backend:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "2. In a separate terminal, start the React frontend:"
echo "   cd resume-optimizer"
echo "   npm run dev"
echo ""
echo "3. Then open your browser to http://localhost:5173"
echo ""
echo "Don't forget to update the OpenAI API key in optimizer.py!"
echo "Replace 'YOUR_OPENAI_API_KEY_HERE' with your actual API key." 