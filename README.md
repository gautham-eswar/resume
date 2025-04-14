# Resume Optimizer

An AI-powered resume optimization system that helps tailor resumes to specific job descriptions using OpenAI's GPT models.

## Features

- Resume parsing (PDF, DOCX, TXT)
- Intelligent keyword extraction from job descriptions
- Semantic matching between resume content and job requirements
- AI-powered resume enhancement
- Modern React frontend with Chakra UI
- RESTful API backend with Flask

## Project Structure

```
.
├── Backend
│   ├── app.py                 # Main Flask application
│   ├── optimizer.py           # Resume optimization pipeline
│   ├── resume_parser.py       # Resume parsing module
│   ├── keyword_extractor.py   # Keyword extraction from job descriptions
│   ├── embeddings.py          # Semantic matching using embeddings
│   ├── enhancer.py           # Resume enhancement with GPT
│   └── PDFsave.py            # PDF handling utilities
│
├── Frontend (resume-optimizer/)
│   ├── src/
│   │   ├── components/       # React components
│   │   └── services/        # API integration services
│   └── public/              # Static assets
│
├── uploads/                 # Temporary storage for uploaded resumes
├── output/                 # Output directory for enhanced resumes
└── requirements.txt        # Python dependencies
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/resume-optimizer.git
cd resume-optimizer
```

2. Set up the Python environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_api_key_here
```

4. Set up the frontend:
```bash
cd resume-optimizer
npm install
```

5. Start the backend server:
```bash
python app.py
```

6. Start the frontend development server:
```bash
cd resume-optimizer
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## API Endpoints

- `POST /api/upload` - Upload and parse a resume
- `POST /api/optimize` - Optimize resume with job description
- `GET /api/download/:resumeId/:format` - Download enhanced resume

## Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key
- `FLASK_ENV` - Set to `development` for debug mode
- `PORT` - Backend server port (default: 5000)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 