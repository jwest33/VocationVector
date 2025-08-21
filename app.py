"""
Flask frontend application for VocationVector job matching system
"""
import os
import json
import subprocess
import threading
import queue
import time
import atexit
import signal
import sys
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import tempfile
import shutil

# Import graph modules and preload heavy dependencies
print("Importing settings...")
from graph.settings import get_settings
print("Settings imported successfully")

# Utility function for parsing JSON fields
def parse_json_field(field_value, default=None):
    """Parse JSON field safely, returning default if parsing fails"""
    if field_value:
        if isinstance(field_value, str):
            try:
                return json.loads(field_value)
            except:
                pass
        elif not isinstance(field_value, (int, float, bool)):
            return field_value
    return default if default is not None else {}

print("Importing LLM server manager...")
from graph.llm_server import LLMServerManager
print("LLM server manager imported successfully")

# Preload heavy modules to avoid delays on first request
print("Preloading heavy modules...")
print("  Loading database module...")
from graph.database import graphDB
print("  Loading embeddings module...")
from graph.embeddings import JobEmbeddings
print("  Loading resume processing module...")
from graph.nodes.resume_processing import ResumeLLMProcessor
print("  Loading pipeline modules...")
from graph.pipeline import run_resume_processing, run_job_processing, run_matching
print("  Initializing embeddings model (this may take a moment)...")
# Set up logger
logger = logging.getLogger(__name__)

# Initialize embeddings singleton at startup
embeddings = JobEmbeddings()
print("All modules preloaded successfully")

print("Creating Flask app...")
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = Path('data/uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}

# Ensure upload folder exists
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

# Initialize SocketIO for real-time updates
# Use threading mode with additional safety options
print("Initializing SocketIO...")
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    logger=False,  # Disable SocketIO logging to reduce noise
    engineio_logger=False,  # Disable Engine.IO logging
    ping_timeout=60,  # Increase ping timeout
    ping_interval=25  # Increase ping interval
)
print("SocketIO initialized successfully")

# Track connected clients
connected_clients = set()

# Store running processes
running_processes = {}
process_outputs = {}

# Progress update queue for thread-safe WebSocket communication
progress_queue = queue.Queue()

# Initialize LLM server manager
llm_manager = None

def start_llm_server():
    """Start the LLM server if configured"""
    global llm_manager
    try:
        settings = get_settings()
        if settings.llm.auto_start_server:
            print("Starting LLM server...")
            llm_manager = LLMServerManager()
            if llm_manager.start():
                print("LLM server started successfully")
                return True
            else:
                print("✗ Failed to start LLM server")
                return False
        else:
            print("LLM auto-start is disabled in settings")
            return False
    except Exception as e:
        print(f"Error starting LLM server: {e}")
        return False

def stop_llm_server():
    """Stop the LLM server if running"""
    global llm_manager
    if llm_manager:
        try:
            print("Stopping LLM server...")
            llm_manager.stop()
            print("LLM server stopped")
        except Exception as e:
            print(f"Error stopping LLM server: {e}")

# Register cleanup handlers
def cleanup():
    """Cleanup function to stop LLM server on exit"""
    stop_llm_server()
    # Also terminate any running pipeline processes
    for process_id, process in running_processes.items():
        try:
            process.terminate()
            print(f"Terminated process: {process_id}")
        except:
            pass

# Register cleanup on exit
atexit.register(cleanup)

# Handle Ctrl+C gracefully
def signal_handler(sig, frame):
    print("\nShutting down...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def emit_progress_update(update_type, data):
    """Thread-safe progress update emission"""
    progress_queue.put({
        'type': update_type,
        'data': data,
        'timestamp': datetime.now().isoformat()
    })

def process_progress_queue():
    """Process queued progress updates - must be called from main thread"""
    while not progress_queue.empty():
        try:
            update = progress_queue.get_nowait()
            socketio.emit(update['type'], update['data'])
        except queue.Empty:
            break
        except Exception as e:
            print(f"Error emitting progress update: {e}")

def run_pipeline_command(command, process_id):
    """Run a pipeline command and stream output to websocket"""
    try:
        print(f"[{process_id}] Starting command: {command}")  # Debug
        
        # Queue initial status
        emit_progress_update('process_status', {
            'process_id': process_id,
            'status': 'starting',
            'message': 'Initializing pipeline...',
            'timestamp': datetime.now().isoformat()
        })
        
        # Set environment to force unbuffered Python output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout and stderr
            text=True,
            bufsize=0,  # No buffering
            env=env,
            cwd=os.getcwd()  # Set working directory
        )
        print(f"[{process_id}] Process started with PID: {process.pid}")  # Debug
        
        running_processes[process_id] = process
        process_outputs[process_id] = []
        
        # Track processing stages
        current_stage = 'initializing'
        last_output_time = time.time()
        
        # Stream output line by line with improved reading
        # Use a separate thread to read output to avoid blocking
        output_queue = queue.Queue()
        
        def read_output(pipe, q):
            """Read output from pipe and put in queue"""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        q.put(line)
                    else:
                        break
            except:
                pass
            finally:
                pipe.close()
        
        # Start output reader thread
        reader_thread = threading.Thread(target=read_output, args=(process.stdout, output_queue))
        reader_thread.daemon = True
        reader_thread.start()
        
        while True:
            # Check if process is still running
            poll = process.poll()
            
            # Try to get output from queue
            try:
                line = output_queue.get(timeout=0.1)
                last_output_time = time.time()
            except queue.Empty:
                # No output available
                if poll is not None:
                    # Process finished - wait a bit for final output
                    time.sleep(0.5)
                    # Get any remaining output
                    while not output_queue.empty():
                        try:
                            line = output_queue.get_nowait()
                            if line and line.strip():
                                emit_progress_update('process_output', {
                                    'process_id': process_id,
                                    'output': line.strip(),
                                    'timestamp': datetime.now().isoformat()
                                })
                        except queue.Empty:
                            break
                    break
                    
                # Send heartbeat if no output for 15 seconds (increased from 5)
                #if time.time() - last_output_time > 15:
                #    emit_progress_update('process_status', {
                #        'process_id': process_id,
                #        'status': 'running',
                #        'message': f'Still {current_stage.replace("_", " ")}...',
                #        'timestamp': datetime.now().isoformat()
                #    })
                #    last_output_time = time.time()
                continue
            
            line_text = line.strip()
            if line_text:
                # Filter out TensorFlow/protobuf warnings and errors
                skip_patterns = [
                    'tensorflow/core/util/port.cc',
                    'oneDNN custom operations',
                    "MessageFactory' object has no attribute",
                    'protobuf',
                    'WARNING:tensorflow',
                    'tf_keras',
                    'I tensorflow',
                    'W tensorflow'
                ]
                
                # Check if this line should be skipped
                should_skip = any(pattern in line_text for pattern in skip_patterns)
                
                if not should_skip:
                    print(f"[{process_id}] Output: {line_text[:100]}")  # Debug log
                
                last_output_time = time.time()
                process_outputs[process_id].append(line_text)
                
                # Skip sending filtered messages to frontend
                if should_skip:
                    continue
                
                # Detect stage transitions and meaningful updates based on output
                stage_update = None
                
                # Resume processing stages
                if 'Loading resume' in line_text or 'Processing resume' in line_text:
                    current_stage = 'processing_resume'
                    stage_update = line_text if 'from' in line_text else 'Processing resume file...'
                elif 'Pass 1' in line_text or 'Pass 2' in line_text:
                    current_stage = 'processing_resume'
                    stage_update = line_text  # Show the actual pass info
                elif 'Skills extracted' in line_text:
                    current_stage = 'processing_resume'
                    stage_update = line_text  # Show skill extraction details
                    
                # Job search stages
                elif 'Starting job search' in line_text or 'Searching for' in line_text:
                    current_stage = 'searching'
                    stage_update = line_text  # Show what we're searching for
                elif 'Opening Google jobs tab' in line_text or 'Launching browser' in line_text:
                    current_stage = 'crawling'
                    stage_update = line_text
                elif 'Crawling' in line_text or 'Google Jobs' in line_text:
                    current_stage = 'crawling'
                    stage_update = 'Crawling job listings...'
                elif 'Found' in line_text and 'job' in line_text.lower():
                    current_stage = 'processing_jobs'
                    stage_update = line_text  # Show how many jobs found
                elif 'Clicking card' in line_text or 'Job' in line_text and '/' in line_text:
                    current_stage = 'processing_jobs'
                    stage_update = line_text  # Show which job we're processing
                elif 'Captured' in line_text and 'characters' in line_text:
                    current_stage = 'processing_jobs'
                    stage_update = line_text  # Show capture details
                    
                # Processing stages
                elif 'Extracting' in line_text or 'extract' in line_text.lower():
                    current_stage = 'extracting'
                    stage_update = line_text if 'job' in line_text.lower() else 'Extracting job details...'
                elif 'Processing job' in line_text and 'template' in line_text:
                    current_stage = 'extracting'
                    stage_update = line_text  # Show template processing
                elif 'Creating embeddings' in line_text or 'embedding' in line_text.lower():
                    current_stage = 'embedding'
                    stage_update = 'Creating embeddings...'
                    
                # Database operations
                elif 'Saving' in line_text or 'Storing' in line_text:
                    current_stage = 'saving'
                    stage_update = line_text if 'jobs' in line_text else 'Saving to database...'
                elif 'Added' in line_text and 'database' in line_text:
                    current_stage = 'saving'
                    stage_update = line_text  # Show what was added
                    
                # Completion and errors
                elif 'Complete' in line_text or 'Finished' in line_text or 'Successfully' in line_text:
                    current_stage = 'completed'
                    stage_update = line_text if len(line_text) < 100 else 'Processing completed!'
                elif 'Error' in line_text or 'Failed' in line_text:
                    current_stage = 'error'
                    stage_update = line_text if len(line_text) < 100 else 'An error occurred'
                
                # Send stage update if detected
                if stage_update:
                    emit_progress_update('process_status', {
                        'process_id': process_id,
                        'status': current_stage,
                        'message': stage_update,
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Always send the raw output
                emit_progress_update('process_output', {
                    'process_id': process_id,
                    'output': line_text,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Determine appropriate message based on process type
        if 'search' in process_id:
            success_msg = 'Job search completed successfully!'
            fail_msg = 'Job search failed'
        elif 'resume' in process_id:
            success_msg = 'Resume processing completed successfully!'
            fail_msg = 'Resume processing failed'
        elif 'match' in process_id:
            success_msg = 'Matching completed successfully!'
            fail_msg = 'Matching failed'
        else:
            success_msg = 'Process completed successfully!'
            fail_msg = 'Process failed'
        
        # Send completion status
        success = return_code == 0
        emit_progress_update('process_complete', {
            'process_id': process_id,
            'return_code': return_code,
            'success': success,
            'message': success_msg if success else fail_msg,
            'timestamp': datetime.now().isoformat(),
            'type': 'Resume processing' if 'resume' in process_id.lower() else 'Job search' if 'search' in process_id else 'Matching' if 'match' in process_id else 'Unknown'
        })
        
        # Clean up
        if process_id in running_processes:
            del running_processes[process_id]
            
    except Exception as e:
        print(f"[{process_id}] Error: {str(e)}")  # Debug
        emit_progress_update('process_error', {
            'process_id': process_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/test')
def test_page():
    """Test upload page"""
    return send_file('test_upload.html')

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    """Test endpoint to verify server is running"""
    print(f"Test endpoint called - Method: {request.method}")
    if request.method == 'POST':
        print(f"POST data - Files: {request.files}")
        print(f"POST data - Form: {request.form}")
        print(f"POST data - JSON: {request.json if request.is_json else 'Not JSON'}")
    return jsonify({
        'success': True,
        'message': 'Server is running',
        'method': request.method,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/llm_status')
def llm_status():
    """Check LLM server status"""
    global llm_manager
    
    if llm_manager:
        is_healthy = llm_manager.is_healthy()
        return jsonify({
            'success': True,
            'running': True,
            'healthy': is_healthy,
            'url': llm_manager.base_url if hasattr(llm_manager, 'base_url') else 'http://localhost:8000'
        })
    else:
        return jsonify({
            'success': True,
            'running': False,
            'healthy': False,
            'message': 'LLM server not initialized'
        })

@app.route('/api/upload_resume', methods=['POST'])
def upload_resume():
    """Handle resume upload"""
    print(f"Upload endpoint called, method: {request.method}")
    print(f"Request files: {request.files}")
    print(f"Request form: {request.form}")
    
    if 'file' not in request.files:
        print("ERROR: No file in request.files")
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    print(f"File object: {file}")
    print(f"Filename: {file.filename}")
    
    if file.filename == '':
        print("ERROR: Empty filename")
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        print(f"Will save to: {filepath}")
        
        # Track if we're replacing an existing file
        replaced = False
        
        # If file exists, delete it first to allow replacement
        if filepath.exists():
            replaced = True
            try:
                filepath.unlink()  # Delete the existing file
                print(f"Replaced existing file: {filename}")
                
                # Also try to delete from database if exists
                from graph.database import graphDB
                db = graphDB()
                resumes = db.get_all_resumes()
                for resume in resumes:
                    if resume.get('filename') == filename:
                        db.delete_resume(resume.get('resume_id'))
                        print(f"Deleted existing database entry for: {filename}")
                        break
            except Exception as e:
                print(f"Warning: Could not clean up existing file: {e}")
        
        try:
            file.save(str(filepath))
            print(f"File saved successfully to: {filepath}")
            
            # Verify the file was saved
            if filepath.exists():
                file_size = filepath.stat().st_size
                print(f"File verified: {filepath} ({file_size} bytes)")
            else:
                print(f"ERROR: File not found after save: {filepath}")
                return jsonify({'success': False, 'error': 'File save verification failed'}), 500
        except Exception as e:
            print(f"ERROR saving file: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'File save failed: {str(e)}'}), 500
        
        # Start processing the resume asynchronously
        process_id = f"resume_{filename.replace('.', '_')}"  # Use filename as ID
        # Use --resume-path for single file processing with unbuffered output
        # Don't auto-start LLM server since Flask app manages it
        command = f'python -u -m graph.pipeline --mode process_resumes --resume-path "{filepath}" --no-auto-server --verbose'
        
        print(f"Starting resume processing: {command}")  # Debug log
        
        thread = threading.Thread(target=run_pipeline_command, args=(command, process_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'process_id': process_id,
            'replaced': replaced
        })
    
    print(f"ERROR: Invalid file type for: {file.filename}")
    return jsonify({'success': False, 'error': 'Invalid file type'}), 400

@app.route('/api/search_jobs', methods=['POST'])
def search_jobs():
    """Start a job search using the streaming crawler"""
    data = request.json
    
    query = data.get('query', '')
    location = data.get('location', 'remote')
    max_jobs = data.get('max_jobs', 20)
    
    if not query:
        return jsonify({'success': False, 'error': 'Query is required'}), 400
    
    # Build command to use streaming pipeline for incremental processing
    process_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Use streaming pipeline for real-time updates
    command = f'python -u -m graph.streaming_pipeline --query "{query}" --location "{location}" --max-jobs {max_jobs}'
    
    # Start the search process
    thread = threading.Thread(target=run_pipeline_command, args=(command, process_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'process_id': process_id,
        'query': query,
        'location': location,
        'max_jobs': max_jobs
    })

@app.route('/api/run_matching', methods=['POST'])
def run_matching():
    """Run matching between resumes and jobs incrementally"""
    data = request.json
    
    # Get optional filters
    resume_filter = data.get('resume_filter', '')
    job_filter = data.get('job_filter', '')
    
    process_id = f"match_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Use streaming matching for incremental results
    command = 'python -u -m graph.streaming_matching --incremental'
    
    if resume_filter:
        command += f' --resume-filter "{resume_filter}"'
    if job_filter:
        command += f' --job-filter "{job_filter}"'
    
    # Start the matching process
    thread = threading.Thread(target=run_pipeline_command, args=(command, process_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'process_id': process_id
    })

@app.route('/api/get_resumes')
def get_resumes():
    """Get list of processed resumes"""
    try:
        from graph.database import graphDB
        import json
        db = graphDB()
        resumes = db.get_all_resumes()
        
        # Format for frontend
        formatted_resumes = []
        
        # Use global parse_json_field utility
        
        for resume in resumes:
            # Parse JSON fields
            experience = parse_json_field(resume.get('experience'), [])
            education = parse_json_field(resume.get('education'), [])
            
            # Parse skills structure which includes categories
            skills_data = parse_json_field(resume.get('skills'), {})
            all_skills = []
            if isinstance(skills_data, dict):
                all_skills.extend(skills_data.get('technical', []))
                all_skills.extend(skills_data.get('languages', []))
                all_skills.extend(skills_data.get('tools', []))
            elif isinstance(skills_data, list):
                all_skills = skills_data
            
            # Parse matching template for more info
            matching_data = parse_json_field(resume.get('matching_template'), {})
            
            # Safely get years_experience, handling None, NaN, or invalid values
            years_exp = resume.get('years_experience')
            if years_exp is None or (isinstance(years_exp, float) and not float('-inf') < years_exp < float('inf')):
                years_exp = 0
            elif isinstance(years_exp, str):
                try:
                    years_exp = float(years_exp)
                except:
                    years_exp = 0
            
            # Get current role from first experience if available
            current_role = ''
            if experience and len(experience) > 0 and isinstance(experience[0], dict):
                current_role = experience[0].get('title', '') or experience[0].get('role', '')
            
            # Parse certifications
            certifications = parse_json_field(resume.get('certifications'), [])
            
            # Parse achievements
            achievements = parse_json_field(resume.get('achievements'), [])
            
            formatted_resumes.append({
                'id': str(resume.get('resume_id', '')),
                'name': resume.get('name', 'Unknown'),
                'email': resume.get('email', ''),
                'phone': resume.get('phone', ''),
                'location': resume.get('location', ''),
                'current_role': current_role,
                'years_experience': years_exp,
                'skills': all_skills[:20],  # Limit for display
                'experience': experience[:5] if experience else [],  # Top 5 experiences
                'education': education[:3] if education else [],  # Top 3 education
                'certifications': certifications[:5] if certifications else [],  # Top 5 certifications
                'achievements': achievements[:5] if achievements else [],  # Top 5 achievements
                'summary': resume.get('summary', ''),
                'full_text': resume.get('full_text', ''),
                'processed_at': resume.get('process_timestamp', ''),
                'filename': resume.get('filename', '')
            })
        
        return jsonify({
            'success': True,
            'resumes': formatted_resumes
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_jobs')
def get_jobs():
    """Get list of crawled jobs"""
    try:
        from graph.database import graphDB
        import json
        db = graphDB()
        jobs = db.get_all_jobs()
        
        # Format for frontend
        formatted_jobs = []
        for job in jobs:
            # Parse JSON fields
            skills = parse_json_field(job.get('skills'), [])
            
            # Safely handle salary fields - they might be None, NaN, or other invalid values
            salary_min = job.get('salary_min')
            salary_max = job.get('salary_max')
            
            # Check for NaN or invalid numeric values
            if salary_min is not None:
                if isinstance(salary_min, (int, float)):
                    if not (float('-inf') < salary_min < float('inf')):
                        salary_min = None
                else:
                    try:
                        salary_min = float(salary_min)
                        if not (float('-inf') < salary_min < float('inf')):
                            salary_min = None
                    except:
                        salary_min = None
            
            if salary_max is not None:
                if isinstance(salary_max, (int, float)):
                    if not (float('-inf') < salary_max < float('inf')):
                        salary_max = None
                else:
                    try:
                        salary_max = float(salary_max)
                        if not (float('-inf') < salary_max < float('inf')):
                            salary_max = None
                    except:
                        salary_max = None
            
            # Parse universal template for additional fields
            vocation_template = parse_json_field(job.get('vocation_template'), {})
            
            formatted_jobs.append({
                'id': str(job.get('job_id', '')),
                'title': job.get('title', 'Unknown Position'),
                'company': job.get('company', 'Unknown Company'),
                'location': job.get('location', ''),
                'salary_min': salary_min,
                'salary_max': salary_max,
                'employment_type': job.get('employment_type', ''),
                'remote_policy': job.get('remote_policy', ''),
                'years_experience_required': job.get('years_experience_required'),
                'skills': skills[:10] if skills else [],
                'responsibilities': parse_json_field(job.get('responsibilities'), [])[:5],
                'qualifications': parse_json_field(job.get('requirements'), [])[:5],
                'benefits': parse_json_field(job.get('benefits'), [])[:5],
                'education_requirements': parse_json_field(job.get('education_requirements'), []),
                'equity': job.get('equity'),
                'bonus': job.get('bonus'),
                'team_size': job.get('team_size'),
                'growth_opportunities': job.get('growth_opportunities'),
                'start_date': job.get('start_date'),
                'full_text': job.get('text', ''),
                'crawled_at': job.get('crawl_timestamp', ''),
                'posted_date': job.get('posted_date', ''),
                'employment_type': job.get('employment_type', ''),
                'via': job.get('via', ''),
                'description': job.get('description', ''),
                # Include some fields from universal template if not in main job
                'culture_fit': vocation_template.get('culture_fit', {}),
                'key_technologies': vocation_template.get('key_technologies', [])
            })
        
        return jsonify({
            'success': True,
            'jobs': formatted_jobs
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_matches')
def get_matches():
    """Get matching results with optional filtering"""
    try:
        from graph.database import graphDB
        db = graphDB()
        
        # Get query parameters for filtering
        min_score = request.args.get('min_score', type=float, default=0.0)
        hidden_resumes = request.args.get('hidden_resumes', '').split(',') if request.args.get('hidden_resumes') else []
        
        # Get more matches to allow for filtering
        matches = db.get_top_matches(limit=500)
        
        # Get all jobs and resumes for lookup
        all_jobs_list = db.get_all_jobs()
        all_resumes_list = db.get_all_resumes()
        
        # Create lookup dictionaries - handle both job_id formats
        all_jobs = {job['job_id']: job for job in all_jobs_list}
        # Also add lookup by job_index for backward compatibility
        for idx, job in enumerate(all_jobs_list):
            all_jobs[str(idx)] = job  # Add by index
            all_jobs[str(job.get('job_index', idx))] = job  # Add by job_index if exists
        
        all_resumes = {resume['resume_id']: resume for resume in all_resumes_list}
        
        # Format for frontend
        formatted_matches = []
        for match in matches:
            # Get job and resume details
            job_id = match.get('job_id', '')
            resume_id = match.get('resume_id', '')
            
            # Look up job details
            job = all_jobs.get(job_id, {})
            job_title = job.get('title', 'Unknown Position')
            company = job.get('company', 'Unknown Company')
            
            # Look up resume details
            resume = all_resumes.get(resume_id, {})
            resume_name = resume.get('name', 'Unknown')
            
            # If name is still unknown, try to get from filename
            if resume_name == 'Unknown' and resume.get('filename'):
                resume_name = resume.get('filename', '').replace('.pdf', '').replace('.docx', '').replace('_', ' ')
            
            # Parse matched skills and gaps
            skills_matched = parse_json_field(match.get('skills_matched', []))
            skills_gap = parse_json_field(match.get('skills_gap', []))
            requirements_matched = parse_json_field(match.get('requirements_matched', []))
            requirements_gap = parse_json_field(match.get('requirements_gap', []))
            
            # Handle NaN values in scores
            def safe_float(value, default=0.0):
                if value is None:
                    return default
                if isinstance(value, float):
                    if not (float('-inf') < value < float('inf')):
                        return default
                try:
                    val = float(value)
                    if not (float('-inf') < val < float('inf')):
                        return default
                    return val
                except:
                    return default
            
            # Get salary info with safe handling
            salary_min = job.get('salary_min')
            salary_max = job.get('salary_max')
            
            # Handle NaN values in salary
            if salary_min is not None and isinstance(salary_min, float):
                if not (float('-inf') < salary_min < float('inf')):
                    salary_min = None
            if salary_max is not None and isinstance(salary_max, float):
                if not (float('-inf') < salary_max < float('inf')):
                    salary_max = None
            
            formatted_matches.append({
                'id': match.get('match_id', ''),
                'resume_id': resume_id,
                'job_id': job_id,
                'resume_name': resume_name,
                'job_title': job_title,
                'company': company,
                'location': job.get('location', ''),
                'posted_date': job.get('posted_date', ''),
                'crawled_at': job.get('crawl_timestamp', ''),
                'salary_min': salary_min,
                'salary_max': salary_max,
                'overall_score': round(safe_float(match.get('overall_score', 0)) * 100),
                'job_fit_score': round(safe_float(match.get('job_fit_score', 0)) * 100) if match.get('job_fit_score') else None,
                'candidate_fit_score': round(safe_float(match.get('candidate_fit_score', 0)) * 100) if match.get('candidate_fit_score') else None,
                'skills_score': round(safe_float(match.get('skills_score', 0)) * 100),
                'experience_score': round(safe_float(match.get('experience_score', 0)) * 100),
                'education_score': round(safe_float(match.get('education_score', 0)) * 100) if match.get('education_score') else None,
                'location_score': round(safe_float(match.get('location_score', 0)) * 100) if match.get('location_score') else None,
                'salary_score': round(safe_float(match.get('salary_score', 0)) * 100) if match.get('salary_score') else None,
                'semantic_score': round(safe_float(match.get('semantic_score', 0)) * 100),
                'title_match_score': round(safe_float(match.get('title_match_score', 0)) * 100),
                'requirements_score': round(safe_float(match.get('requirements_score', 0)) * 100),
                'summary_to_description_score': round(safe_float(match.get('summary_to_description_score', 0)) * 100) if match.get('summary_to_description_score') else None,
                'experience_to_requirements_score': round(safe_float(match.get('experience_to_requirements_score', 0)) * 100) if match.get('experience_to_requirements_score') else None,
                'skills_to_skills_score': round(safe_float(match.get('skills_to_skills_score', 0)) * 100) if match.get('skills_to_skills_score') else None,
                'llm_score': round(safe_float(match.get('llm_score', 0)) * 100),
                'llm_assessment': match.get('llm_assessment', ''),  # Contains the reasoning text and recommendations
                'llm_recommendations': [],  # Extracted from llm_assessment in frontend if needed
                'skills_matched': skills_matched,
                'skills_gap': skills_gap,
                'exceeded_skills': parse_json_field(match.get('exceeded_skills', [])),
                'requirements_matched': requirements_matched,
                'requirements_gap': requirements_gap,
                'education_gaps': parse_json_field(match.get('education_gaps', [])),
                'experience_gaps': parse_json_field(match.get('experience_gaps', {})),
                'salary_match': parse_json_field(match.get('salary_match', {})),
                'location_preference_met': match.get('location_preference_met'),
                'remote_preference_met': match.get('remote_preference_met'),
                'match_reasons': parse_json_field(match.get('match_reasons', [])),
                'confidence_score': round(match.get('confidence_score', 0) * 100) if match.get('confidence_score') else None,
                'created_at': match.get('match_timestamp', '')
            })
        
        # Apply filters
        filtered_matches = []
        for match in formatted_matches:
            # Filter by minimum score
            if match['overall_score'] < min_score * 100:  # Convert to percentage
                continue
            
            # Filter by hidden resumes
            if match['resume_id'] in hidden_resumes:
                continue
            
            filtered_matches.append(match)
        
        # Sort by overall score
        filtered_matches.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'matches': filtered_matches,
            'total_before_filter': len(formatted_matches),
            'total_after_filter': len(filtered_matches)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save_preferences', methods=['POST'])
def save_preferences():
    """Save job seeker preferences"""
    data = request.json
    
    try:
        # Save to a preferences file
        prefs_file = Path('data/preferences.json')
        prefs_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(prefs_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Preferences saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_preferences')
def get_preferences():
    """Get saved preferences"""
    try:
        prefs_file = Path('data/preferences.json')
        if prefs_file.exists():
            with open(prefs_file, 'r') as f:
                preferences = json.load(f)
        else:
            # Return default preferences
            preferences = {
                'preferred_titles': ['Data Engineer', 'ML Engineer'],
                'locations': ['Remote', 'Denver, CO', 'Austin, TX'],
                'min_salary': 150000,
                'max_salary': 190000,
                'work_type': 'full-time',
                'domains': ['Healthcare', 'Climate', 'AI'],
                'weights': {
                    'skills': 40,
                    'experience': 20,
                    'location': 15,
                    'domain': 15,
                    'compensation': 10
                }
            }
        
        return jsonify({'success': True, 'preferences': preferences})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_job/<job_id>')
def get_job_detail(job_id):
    """Get detailed job information"""
    try:
        from graph.database import graphDB
        import json
        db = graphDB()
        job = db.get_job_by_id(job_id)
        
        if not job:
            return jsonify({'success': False, 'error': 'Job not found'}), 404
        
        # Ensure job is a dictionary
        if isinstance(job, str):
            # If somehow we got a string, try to parse it
            try:
                job = json.loads(job)
            except:
                return jsonify({'success': False, 'error': 'Invalid job data format'}), 500
        
        if not isinstance(job, dict):
            return jsonify({'success': False, 'error': f'Expected dict, got {type(job).__name__}'}), 500
        
        # Parse JSON fields
        # Use global parse_json_field utility
        
        # Check for NaN values in salary fields
        salary_min = job.get('salary_min')
        salary_max = job.get('salary_max')
        
        # Convert NaN to None for proper JSON serialization
        if salary_min is None or (isinstance(salary_min, float) and not float('-inf') < salary_min < float('inf')):
            salary_min = None
        if salary_max is None or (isinstance(salary_max, float) and not float('-inf') < salary_max < float('inf')):
            salary_max = None
        
        # Parse universal template if present for additional fields
        # The vocation_template field is stored as a JSON string in the database
        vocation_template = parse_json_field(job.get('vocation_template'), {})
        
        # Handle both nested and flat universal template structures
        # The streaming pipeline creates a flat structure, while batch pipeline may have nested
        if 'compensation' in vocation_template:
            # Nested structure (batch pipeline)
            compensation = vocation_template.get('compensation', {})
            culture_fit = vocation_template.get('culture_fit', {})
            location_info = vocation_template.get('location', {})
            metadata = vocation_template.get('metadata', {})
        else:
            # Flat structure (streaming pipeline) - create nested structure from flat fields
            compensation = {
                'minimum_salary': vocation_template.get('salary_min'),
                'maximum_salary': vocation_template.get('salary_max'),
                'currency': vocation_template.get('salary_currency', 'USD'),
                'benefits_required': vocation_template.get('benefits', []),
                'equity_expectation': vocation_template.get('equity'),
                'bonus_structure': vocation_template.get('bonus')
            }
            culture_fit = {
                'team_size_preference': vocation_template.get('team_size'),
                'career_growth_importance': vocation_template.get('growth_opportunities'),
                'work_life_balance_preference': '',
                'management_style_preference': ''
            }
            location_info = {
                'work_arrangement': vocation_template.get('remote_policy'),
                'visa_sponsorship_needed': vocation_template.get('visa_sponsorship'),
                'relocation_willing': vocation_template.get('relocation_assistance')
            }
            metadata = {
                'company': vocation_template.get('company', job.get('company', '')),
                'department': '',
                'industry': ''
            }
        
        # Build proper description from universal template if needed
        description = job.get('description', '')
        if not description or description == 'None':
            description_parts = []
            # Try job_summary first, then summary
            if vocation_template.get('job_summary'):
                description_parts.append(vocation_template.get('job_summary'))
            elif vocation_template.get('summary'):
                description_parts.append(vocation_template.get('summary'))
            
            # Check for responsibilities (streaming) or key_responsibilities (batch)
            responsibilities_field = vocation_template.get('responsibilities') or vocation_template.get('key_responsibilities', [])
            if responsibilities_field:
                description_parts.append("\n\nKey Responsibilities:")
                for resp in responsibilities_field[:5]:
                    if resp:
                        description_parts.append(f"• {resp}")
            description = '\n'.join(description_parts) if description_parts else job.get('text', '')[:500]
        
        # Extract requirements from universal template if needed
        requirements = parse_json_field(job.get('requirements'), [])
        if not requirements or (len(requirements) == 1 and requirements[0] == 'No specific requirements listed'):
            requirements = []
            # Add experience requirements
            for exp_req in vocation_template.get('experience_requirements', []):
                if isinstance(exp_req, dict) and exp_req.get('description'):
                    requirements.append(exp_req['description'])
            # Add education requirements
            for edu_req in vocation_template.get('education_requirements', []):
                if isinstance(edu_req, dict):
                    degree = edu_req.get('degree_level', '')
                    field = edu_req.get('field_of_study', '')
                    if degree:
                        req_text = f"{degree}"
                        if field:
                            req_text += f" in {field}"
                        requirements.append(req_text)
        
        # Extract responsibilities from universal template
        responsibilities = parse_json_field(job.get('responsibilities'), [])
        if not responsibilities or (len(responsibilities) == 1 and responsibilities[0] == 'See job description'):
            # Try both fields - streaming uses 'responsibilities', batch uses 'key_responsibilities'
            responsibilities = vocation_template.get('responsibilities') or vocation_template.get('key_responsibilities', [])
        
        # Extract benefits from universal template
        benefits = parse_json_field(job.get('benefits'), [])
        if not benefits or (len(benefits) == 1 and benefits[0] == 'Benefits package available'):
            benefits = compensation.get('benefits_required', [])
        
        # Extract skills properly
        skills = parse_json_field(job.get('skills'), [])
        technical_skills = []
        for skill in vocation_template.get('technical_skills', []):
            if isinstance(skill, dict):
                skill_info = {
                    'name': skill.get('skill_name', skill.get('skill', '')),
                    'proficiency': skill.get('required_proficiency', ''),
                    'years': skill.get('years_required'),
                    'mandatory': skill.get('is_mandatory', False)
                }
                technical_skills.append(skill_info)
                # Also add skill name to basic skills list if not present
                if skill_info['name'] and skill_info['name'] not in skills:
                    skills.append(skill_info['name'])
            elif isinstance(skill, str):
                # Handle simple string skills
                technical_skills.append({
                    'name': skill,
                    'proficiency': '',
                    'years': None,
                    'mandatory': False
                })
                if skill and skill not in skills:
                    skills.append(skill)
        
        # Extract key technologies
        key_technologies = []
        for tool in vocation_template.get('tools_technologies', []):
            if isinstance(tool, dict):
                tech_name = tool.get('skill_name', tool.get('name', ''))
                if tech_name:
                    key_technologies.append(tech_name)
            elif isinstance(tool, str):
                key_technologies.append(tool)
        
        # Extract certifications
        certifications = vocation_template.get('certifications', [])
        
        # Parse apply links
        apply_links = parse_json_field(job.get('apply_links'), [])
        
        formatted_job = {
            'id': job.get('job_id', ''),
            'title': job.get('title', '') or vocation_template.get('title', 'Unknown Position'),
            'company': job.get('company', '') or metadata.get('company', 'Unknown Company'),
            'location': job.get('location', ''),
            'apply_links': apply_links,  # Add apply links
            'salary_min': salary_min or compensation.get('minimum_salary'),
            'salary_max': salary_max or compensation.get('maximum_salary'),
            'salary_currency': compensation.get('currency', 'USD'),
            'employment_type': job.get('employment_type', '') or vocation_template.get('employment_type', ''),
            'remote_policy': job.get('remote_policy', '') or location_info.get('work_arrangement', ''),
            'description': description,
            'requirements': requirements,
            'responsibilities': responsibilities,
            'benefits': benefits,
            'skills': skills,
            'technical_skills': technical_skills,  # Detailed skill info
            'key_technologies': key_technologies,
            'certifications': certifications,
            'education_requirements': vocation_template.get('education_requirements', []),
            'years_experience_required': job.get('years_experience_required') or vocation_template.get('total_years_experience'),
            'management_experience': vocation_template.get('management_experience'),
            'equity': job.get('equity') or compensation.get('equity_expectation', ''),
            'bonus': job.get('bonus') or compensation.get('bonus_structure', ''),
            'team_size': job.get('team_size') or culture_fit.get('team_size_preference', ''),
            'growth_opportunities': job.get('growth_opportunities') or culture_fit.get('career_growth_importance', ''),
            'work_life_balance': culture_fit.get('work_life_balance_preference', ''),
            'management_style': culture_fit.get('management_style_preference', ''),
            'preferred_industries': vocation_template.get('preferred_industries', []),
            'soft_skills': [s.get('skill_name', '') if isinstance(s, dict) else str(s) for s in vocation_template.get('soft_skills', [])],
            'start_date': job.get('start_date') or vocation_template.get('start_date', ''),
            'visa_sponsorship': location_info.get('visa_sponsorship_needed'),
            'relocation_assistance': location_info.get('relocation_willing'),
            'department': metadata.get('department', ''),
            'industry': metadata.get('industry', ''),
            'full_text': job.get('text', ''),
            'crawled_at': job.get('crawl_timestamp', ''),
            'posted_date': job.get('posted_date', ''),
            'via': job.get('via', ''),
            'search_query': job.get('search_query', ''),
            'search_location': job.get('search_location', ''),
            'job_index': job.get('job_index', 0),
            'extraction_confidence': vocation_template.get('extraction_confidence', 0),
            # Include the full universal template for any fields we might have missed
            'vocation_template': vocation_template
        }
        
        return jsonify({'success': True, 'job': formatted_job})
    except Exception as e:
        import traceback
        print(f"Error in get_job_detail for job_id {job_id}:")
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_resume/<resume_id>')
def get_resume_detail(resume_id):
    """Get detailed resume information"""
    try:
        from graph.database import graphDB
        import json
        db = graphDB()
        resume = db.get_resume_by_id(resume_id)
        
        if not resume:
            return jsonify({'success': False, 'error': 'Resume not found'}), 404
        
        # Parse JSON fields
        # Use global parse_json_field utility
        
        # Parse skills structure which includes categories
        skills_data = parse_json_field(resume.get('skills'), {})
        all_skills = []
        if isinstance(skills_data, dict):
            all_skills.extend(skills_data.get('technical', []))
            all_skills.extend(skills_data.get('languages', []))
            all_skills.extend(skills_data.get('tools', []))
        elif isinstance(skills_data, list):
            all_skills = skills_data
        
        formatted_resume = {
            'id': resume.get('resume_id', ''),
            'name': resume.get('name', 'Unknown'),
            'email': resume.get('email', ''),
            'phone': resume.get('phone', ''),
            'location': resume.get('location', ''),
            'linkedin': resume.get('linkedin', ''),
            'github': resume.get('github', ''),
            'years_experience': resume.get('years_experience', 0),
            'summary': resume.get('summary', ''),
            'experience': parse_json_field(resume.get('experience')),
            'education': parse_json_field(resume.get('education')),
            'skills': all_skills,
            'skills_categorized': skills_data if isinstance(skills_data, dict) else {'all': all_skills},
            'certifications': parse_json_field(resume.get('certifications')),
            'achievements': parse_json_field(resume.get('achievements')),
            'salary_expectations': parse_json_field(resume.get('salary_expectations'), {}),
            'work_preferences': parse_json_field(resume.get('work_preferences'), {}),
            'full_text': resume.get('full_text', ''),
            'processed_at': resume.get('process_timestamp', ''),
            'filename': resume.get('filename', ''),
            'matching_template': parse_json_field(resume.get('matching_template'), {}),
            'vocation_template': parse_json_field(resume.get('vocation_template'), {})
        }
        
        return jsonify({'success': True, 'resume': formatted_resume})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/update_resume/<resume_id>', methods=['POST'])
def update_resume(resume_id):
    """Update resume details in the database"""
    try:
        from graph.database import graphDB
        import json
        
        db = graphDB()
        
        # Get the existing resume
        resume = db.get_resume_by_id(resume_id)
        if not resume:
            return jsonify({'success': False, 'error': 'Resume not found'}), 404
        
        # Get updated data
        updated_data = request.json
        
        # Update the resume fields
        resume['name'] = updated_data.get('name', resume.get('name'))
        resume['email'] = updated_data.get('email', resume.get('email'))
        resume['phone'] = updated_data.get('phone', resume.get('phone'))
        resume['location'] = updated_data.get('location', resume.get('location'))
        resume['linkedin'] = updated_data.get('linkedin', resume.get('linkedin'))
        resume['github'] = updated_data.get('github', resume.get('github'))
        resume['summary'] = updated_data.get('summary', resume.get('summary'))
        resume['years_experience'] = updated_data.get('years_experience', resume.get('years_experience'))
        
        # Update JSON fields
        if 'salary_expectations' in updated_data:
            resume['salary_expectations'] = json.dumps(updated_data['salary_expectations'])
        
        if 'work_preferences' in updated_data:
            resume['work_preferences'] = json.dumps(updated_data['work_preferences'])
        
        if 'certifications' in updated_data:
            resume['certifications'] = json.dumps(updated_data['certifications'])
        
        if 'achievements' in updated_data:
            resume['achievements'] = json.dumps(updated_data['achievements'])
        
        # Re-add the resume with updated data
        # Note: LanceDB will version the data automatically
        db.add_resume(resume)
        
        return jsonify({'success': True, 'message': 'Resume updated successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/update_skills/<resume_id>', methods=['POST'])
def update_skills(resume_id):
    """Update skills for a specific resume"""
    try:
        from graph.database import graphDB
        import json
        
        db = graphDB()
        resume = db.get_resume_by_id(resume_id)
        
        if not resume:
            return jsonify({'success': False, 'error': 'Resume not found'}), 404
        
        skills = request.json.get('skills', [])
        
        # Update skills in the resume
        skills_data = {
            'technical': [],
            'languages': [],
            'tools': []
        }
        
        for skill in skills:
            if isinstance(skill, dict):
                skill_name = skill.get('skill_name', '')
                # Categorize skills (simple heuristic)
                if any(lang in skill_name.lower() for lang in ['python', 'java', 'javascript', 'c++', 'ruby', 'go', 'rust']):
                    skills_data['languages'].append(skill_name)
                elif any(tool in skill_name.lower() for tool in ['docker', 'kubernetes', 'aws', 'git', 'jenkins']):
                    skills_data['tools'].append(skill_name)
                else:
                    skills_data['technical'].append(skill_name)
            else:
                skills_data['technical'].append(skill)
        
        resume['skills'] = json.dumps(skills_data)
        
        # Re-add the resume with updated skills
        db.add_resume(resume)
        
        return jsonify({'success': True, 'message': 'Skills updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_analytics')
def get_analytics():
    """Get analytics data"""
    try:
        from graph.database import graphDB
        db = graphDB()
        
        # Get counts
        resumes_count = len(db.get_all_resumes())
        jobs_count = len(db.get_all_jobs())
        matches = db.get_top_matches(limit=100)
        matches_count = len(matches)
        
        # Calculate average match score
        avg_score = 0
        if matches:
            avg_score = sum(m.get('overall_score', 0) for m in matches) / len(matches) * 100
        
        # Get top skills from jobs
        all_skills = []
        for job in db.get_all_jobs():
            all_skills.extend(job.get('extracted_data', {}).get('skills', []))
        
        # Count skill frequencies
        skill_counts = {}
        for skill in all_skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Get top 10 skills
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analytics = {
            'resumes_parsed': resumes_count,
            'jobs_crawled': jobs_count,
            'matches_found': matches_count,
            'avg_match_score': round(avg_score),
            'top_skills': [{'skill': s[0], 'count': s[1]} for s in top_skills]
        }
        
        return jsonify({'success': True, 'analytics': analytics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate_analytics_report', methods=['POST'])
def generate_analytics_report():
    """Generate comprehensive analytics report using LLM for a specific resume"""
    try:
        from graph.database import graphDB
        from openai import OpenAI
        import json
        import os
        
        # Get resume_id from request
        data = request.json
        resume_id = data.get('resume_id')
        
        print(f"Received analytics report request with data: {data}")
        print(f"Resume ID: {resume_id}")
        
        if not resume_id:
            return jsonify({'success': False, 'error': 'Resume ID is required'}), 400
        
        db = graphDB()
        
        # Verify resume exists
        resume = db.get_resume_by_id(resume_id)
        if not resume:
            print(f"Resume not found for ID: {resume_id}")
            return jsonify({'success': False, 'error': f'Resume not found with ID: {resume_id}'}), 404
        
        print(f"Found resume: {resume.get('filename', 'Unknown')}")
        
        # Load user preferences
        prefs_file = Path('data/preferences.json')
        if prefs_file.exists():
            with open(prefs_file, 'r') as f:
                user_preferences = json.load(f)
        else:
            # Use default preferences
            user_preferences = {
                'preferred_titles': 'Not specified',
                'locations': 'Not specified',
                'min_salary': 'Not specified',
                'max_salary': 'Not specified',
                'work_type': 'Not specified',
                'domains': 'Not specified',
                'weights': {
                    'skills': 40,
                    'experience': 20,
                    'location': 15,
                    'domain': 15,
                    'compensation': 10
                }
            }
        
        # Gather all data
        resumes = db.get_all_resumes()
        jobs = db.get_all_jobs()
        matches = db.get_top_matches(limit=100)
        
        # Prepare data for analysis
        # Extract skills from all resumes (skills are stored as JSON strings in the database)
        all_resume_skills = []
        resume_experiences = []
        resume_educations = []
        for resume in resumes:
            # The 'skills' field in the database is a JSON string
            if resume.get('skills'):
                try:
                    if isinstance(resume['skills'], str) and resume['skills'].strip():
                        skills_data = json.loads(resume['skills'])
                        if isinstance(skills_data, list):
                            all_resume_skills.extend(skills_data)
                        elif isinstance(skills_data, dict):
                            # May have categories like technical, soft skills
                            for category, skill_list in skills_data.items():
                                if isinstance(skill_list, list):
                                    all_resume_skills.extend(skill_list)
                    elif isinstance(resume['skills'], list):
                        all_resume_skills.extend(resume['skills'])
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to parse skills for resume {resume.get('resume_id')}: {e}")
            
            # Also check experience field (JSON string)
            if resume.get('experience'):
                try:
                    if isinstance(resume['experience'], str) and resume['experience'].strip():
                        exp_data = json.loads(resume['experience'])
                        if isinstance(exp_data, list):
                            resume_experiences.extend(exp_data)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            
            # Also check education field (JSON string)
            if resume.get('education'):
                try:
                    if isinstance(resume['education'], str) and resume['education'].strip():
                        edu_data = json.loads(resume['education'])
                        if isinstance(edu_data, list):
                            resume_educations.extend(edu_data)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
        
        # Extract skills and requirements from all jobs (stored as JSON strings in database)
        all_job_skills = []
        all_job_requirements = []
        job_titles = []
        salary_ranges = []
        
        for job in jobs:
            # The 'skills' field in the database is a JSON string
            if job.get('skills'):
                try:
                    if isinstance(job['skills'], str) and job['skills'].strip():
                        skills_data = json.loads(job['skills'])
                        if isinstance(skills_data, list):
                            all_job_skills.extend(skills_data)
                    elif isinstance(job['skills'], list):
                        all_job_skills.extend(job['skills'])
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to parse skills for job {job.get('job_id')}: {e}")
            
            # Get requirements (also JSON string)
            if job.get('requirements'):
                try:
                    if isinstance(job['requirements'], str) and job['requirements'].strip():
                        req_data = json.loads(job['requirements'])
                        if isinstance(req_data, list):
                            all_job_requirements.extend(req_data)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            
            if job.get('title'):
                job_titles.append(job.get('title'))
            
            # Check for salary info
            if job.get('salary_min') or job.get('salary_max'):
                salary_ranges.append({
                    'min': job.get('salary_min'),
                    'max': job.get('salary_max')
                })
        
        # Count frequencies
        from collections import Counter
        resume_skill_counts = Counter(all_resume_skills)
        job_skill_counts = Counter(all_job_skills)
        title_counts = Counter(job_titles)
        
        # Find skill gaps
        top_job_skills = set([skill for skill, _ in job_skill_counts.most_common(20)]) if job_skill_counts else set()
        top_resume_skills = set([skill for skill, _ in resume_skill_counts.most_common(20)]) if resume_skill_counts else set()
        skill_gaps = list(top_job_skills - top_resume_skills) if top_job_skills else []
        skill_strengths = list(top_resume_skills & top_job_skills) if (top_resume_skills and top_job_skills) else []
        
        # If no skills were extracted, provide a message
        if not all_job_skills and not all_resume_skills:
            skill_message = "Skills data not available - consider reprocessing jobs and resumes with skill extraction enabled"
        else:
            skill_message = None
        
        # Calculate match statistics
        match_scores = [m.get('overall_score', 0) for m in matches]
        avg_score = sum(match_scores) / len(match_scores) * 100 if match_scores else 0
        
        # Prepare prompt for LLM analysis
        analysis_prompt = f"""Analyze the following job matching data and provide comprehensive insights:

## Data Summary:
- Total Resumes Analyzed: {len(resumes)}
- Total Jobs Analyzed: {len(jobs)}
- Total Matches: {len(matches)}
- Average Match Score: {avg_score:.1f}%

## User Search Preferences:
- Preferred Job Titles: {user_preferences.get('preferred_titles', 'Not specified')}
- Preferred Locations: {user_preferences.get('locations', 'Not specified')}
- Salary Range: ${user_preferences.get('min_salary', 'Not specified')} - ${user_preferences.get('max_salary', 'Not specified')}
- Work Type: {user_preferences.get('work_type', 'Not specified')}
- Preferred Domains: {user_preferences.get('domains', 'Not specified')}

## Matching Weight Configuration:
- Skills Match: {user_preferences.get('weights', {}).get('skills', 40)}%
- Experience Level: {user_preferences.get('weights', {}).get('experience', 20)}%
- Location Fit: {user_preferences.get('weights', {}).get('location', 15)}%
- Domain Relevance: {user_preferences.get('weights', {}).get('domain', 15)}%
- Compensation Fit: {user_preferences.get('weights', {}).get('compensation', 10)}%

## Top Skills in Job Market (Most Demanded):
{', '.join([f"{skill} ({count})" for skill, count in job_skill_counts.most_common(15)])}

## Top Skills in Resume Pool:
{', '.join([f"{skill} ({count})" for skill, count in resume_skill_counts.most_common(15)])}

## Critical Skill Gaps (high demand, low supply):
{', '.join(skill_gaps[:10]) if skill_gaps else 'None identified'}

## Strong Skill Alignments:
{', '.join(skill_strengths[:10]) if skill_strengths else 'None identified'}

## Top Job Titles:
{', '.join([f"{title} ({count})" for title, count in title_counts.most_common(10)])}

## Match Score Distribution:
- Excellent Matches (80-100%): {len([s for s in match_scores if s >= 0.8])}
- Good Matches (60-79%): {len([s for s in match_scores if 0.6 <= s < 0.8])}
- Fair Matches (40-59%): {len([s for s in match_scores if 0.4 <= s < 0.6])}
- Poor Matches (<40%): {len([s for s in match_scores if s < 0.4])}

Based on this data, provide analysis in TWO sections only:

1. **Market Insights**: Key trends, patterns, and observations about the job market and how it aligns with user preferences

2. **Recommendations**: Actionable advice for improving match rates, including both resume improvements and search preference adjustments

Keep responses concise and practical. Consider whether low match scores are due to resume gaps vs preference-market misalignment."""

        # Get LLM analysis
        llm_client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key")
        )
        
        # Make LLM call
        response = llm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen3-4b-instruct-2507-f16"),
            messages=[
                {"role": "system", "content": "You are an expert analyst providing job market insights. Return your analysis as plain text, not JSON."},
                {"role": "user", "content": analysis_prompt + "\n\nProvide your response as plain text sections, not JSON format."}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        llm_response = response.choices[0].message.content
        
        # Parse the response to extract Market Insights and Recommendations
        analysis = {}
        
        # Look for section markers
        market_markers = ["Market Insights", "MARKET INSIGHTS", "1. Market Insights", "**Market Insights**"]
        rec_markers = ["Recommendations", "RECOMMENDATIONS", "2. Recommendations", "**Recommendations**"]
        
        # Find Market Insights section
        market_start = -1
        for marker in market_markers:
            idx = llm_response.find(marker)
            if idx != -1:
                market_start = idx + len(marker)
                # Skip past colon and whitespace
                while market_start < len(llm_response) and llm_response[market_start] in [':', ' ', '\n', '*']:
                    market_start += 1
                break
        
        # Find Recommendations section
        rec_start = -1
        for marker in rec_markers:
            idx = llm_response.find(marker)
            if idx != -1:
                rec_start = idx + len(marker)
                # Skip past colon and whitespace
                while rec_start < len(llm_response) and llm_response[rec_start] in [':', ' ', '\n', '*']:
                    rec_start += 1
                break
        
        # Extract sections
        if market_start != -1 and rec_start != -1:
            if market_start < rec_start:
                # Market insights comes first
                analysis["market_insights"] = llm_response[market_start:rec_start].strip()
                # Remove the Recommendations header from the end
                for marker in rec_markers:
                    if analysis["market_insights"].endswith(marker):
                        analysis["market_insights"] = analysis["market_insights"][:-len(marker)].strip()
                
                analysis["recommendations"] = llm_response[rec_start:].strip()
            else:
                # Recommendations comes first
                analysis["recommendations"] = llm_response[rec_start:market_start].strip()
                # Remove the Market Insights header from the end
                for marker in market_markers:
                    if analysis["recommendations"].endswith(marker):
                        analysis["recommendations"] = analysis["recommendations"][:-len(marker)].strip()
                
                analysis["market_insights"] = llm_response[market_start:].strip()
        elif market_start != -1:
            # Only found market insights
            analysis["market_insights"] = llm_response[market_start:].strip()
            analysis["recommendations"] = "No specific recommendations provided."
        elif rec_start != -1:
            # Only found recommendations
            analysis["recommendations"] = llm_response[rec_start:].strip()
            analysis["market_insights"] = "No market insights provided."
        else:
            # No clear sections found, split the response
            mid_point = len(llm_response) // 2
            analysis["market_insights"] = llm_response[:mid_point].strip()
            analysis["recommendations"] = llm_response[mid_point:].strip()
        
        # Clean up any remaining markdown formatting while preserving structure
        for key in ["market_insights", "recommendations"]:
            if key in analysis:
                # Remove leading/trailing asterisks but keep internal formatting
                analysis[key] = analysis[key].strip('*').strip()
        
        # Compile final report
        report = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'resume_id': resume_id,
            'resume_name': resume.get('filename', 'Unknown'),
            'statistics': {
                'resumes_count': len(resumes),
                'jobs_count': len(jobs),
                'matches_count': len(matches),
                'avg_match_score': round(avg_score, 1),
                'match_distribution': {
                    'excellent': len([s for s in match_scores if s >= 0.8]),
                    'good': len([s for s in match_scores if 0.6 <= s < 0.8]),
                    'fair': len([s for s in match_scores if 0.4 <= s < 0.6]),
                    'poor': len([s for s in match_scores if s < 0.4])
                }
            },
            'skills_analysis': {
                'top_demanded_skills': [{'skill': s, 'count': c} for s, c in job_skill_counts.most_common(15)] if job_skill_counts else [],
                'top_resume_skills': [{'skill': s, 'count': c} for s, c in resume_skill_counts.most_common(15)] if resume_skill_counts else [],
                'skill_gaps': skill_gaps[:10] if skill_gaps else [],
                'skill_strengths': skill_strengths[:10] if skill_strengths else [],
                'message': skill_message
            },
            'market_analysis': {
                'top_job_titles': [{'title': t, 'count': c} for t, c in title_counts.most_common(10)]
            },
            'llm_analysis': analysis
        }
        
        # Save report to database
        report_data = {
            'resume_id': resume_id,
            'report_timestamp': datetime.now().isoformat(),
            'generation_date': datetime.now().strftime("%Y-%m-%d"),
            'total_jobs_analyzed': len(jobs),
            'total_matches_analyzed': len(matches),
            'avg_match_score': round(avg_score, 1),
            'match_distribution': json.dumps(report['statistics']['match_distribution']),
            'top_demanded_skills': json.dumps(report['skills_analysis']['top_demanded_skills']),
            'skill_gaps': json.dumps(report['skills_analysis']['skill_gaps']),
            'skill_strengths': json.dumps(report['skills_analysis']['skill_strengths']),
            'top_job_titles': json.dumps(report['market_analysis']['top_job_titles']),
            'market_insights': analysis.get('market_insights', ''),
            'recommendations': analysis.get('recommendations', ''),
            'user_preferences': json.dumps(user_preferences),
            'full_report_json': json.dumps(report)
        }
        
        saved = db.save_analytics_report(report_data)
        if saved:
            logger.info(f"Analytics report saved for resume {resume_id}")
        else:
            logger.warning(f"Failed to save analytics report for resume {resume_id}")
        
        return jsonify(report)
        
    except Exception as e:
        print(f"Error generating analytics report: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_analytics_report/<resume_id>')
def get_analytics_report(resume_id):
    """Get existing analytics report for a resume"""
    try:
        from graph.database import graphDB
        db = graphDB()
        
        report = db.get_analytics_report(resume_id)
        if report:
            # Parse JSON fields
            if report.get('full_report_json'):
                try:
                    full_report = json.loads(report['full_report_json'])
                    # Ensure it's wrapped in success structure
                    if 'success' in full_report:
                        return jsonify(full_report)
                    else:
                        # Wrap the report data properly
                        return jsonify({
                            'success': True,
                            'report': full_report
                        })
                except Exception as e:
                    print(f"Error parsing full_report_json: {e}")
                    pass
            
            # Return basic report data if full report not available
            return jsonify({
                'success': True,
                'exists': True,
                'report': report
            })
        else:
            return jsonify({
                'success': True,
                'exists': False
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/list_analytics_reports')
def list_analytics_reports():
    """List all analytics reports with basic info"""
    try:
        from graph.database import graphDB
        db = graphDB()
        
        reports = db.get_all_analytics_reports()
        
        # Get resume details for each report
        report_list = []
        for report in reports:
            resume = db.get_resume_by_id(report['resume_id'])
            report_list.append({
                'resume_id': report['resume_id'],
                'resume_name': resume.get('filename', 'Unknown') if resume else 'Unknown',
                'generation_date': report.get('generation_date'),
                'avg_match_score': report.get('avg_match_score'),
                'total_jobs_analyzed': report.get('total_jobs_analyzed'),
                'total_matches_analyzed': report.get('total_matches_analyzed')
            })
        
        return jsonify({
            'success': True,
            'reports': report_list
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop_process/<process_id>', methods=['POST'])
def stop_process(process_id):
    """Stop a running process"""
    if process_id in running_processes:
        try:
            process = running_processes[process_id]
            process.terminate()
            del running_processes[process_id]
            return jsonify({'success': True, 'message': 'Process stopped'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'Process not found'}), 404

@app.route('/api/delete_resume/<resume_id>', methods=['DELETE'])
def delete_resume(resume_id):
    """Delete a resume from the database, filesystem, and cascade delete associated matches"""
    try:
        from graph.database import graphDB
        db = graphDB()
        
        # First get the resume to find the filename
        resumes = db.get_all_resumes()
        resume_to_delete = None
        for resume in resumes:
            if resume.get('resume_id') == resume_id:
                resume_to_delete = resume
                break
        
        # Delete from filesystem if found
        if resume_to_delete and resume_to_delete.get('filename'):
            filename = resume_to_delete.get('filename')
            filepath = app.config['UPLOAD_FOLDER'] / filename
            if filepath.exists():
                try:
                    filepath.unlink()
                    print(f"Deleted file from filesystem: {filepath}")
                except Exception as e:
                    print(f"Warning: Could not delete file {filepath}: {e}")
        
        # Count matches that will be deleted (for reporting)
        matches_before = len(db.get_top_matches(limit=1000))
        
        # Delete from database (this now cascades to matches)
        success = db.delete_resume(resume_id)
        
        if success:
            # Count how many matches were deleted
            matches_after = len(db.get_top_matches(limit=1000))
            matches_deleted = matches_before - matches_after
            
            message = 'Resume deleted successfully'
            if matches_deleted > 0:
                message += f' (and {matches_deleted} associated matches)'
                
            return jsonify({'success': True, 'message': message, 'matches_deleted': matches_deleted})
        else:
            return jsonify({'success': False, 'error': 'Failed to delete resume'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete_job/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job from the database and cascade delete associated matches"""
    try:
        from graph.database import graphDB
        db = graphDB()
        
        # Note: Jobs are stored in bulk JSON files (multiple jobs per file)
        # so we can't delete individual job files. We only delete from database.
        # The bulk JSON files in data/jobs/ contain crawl history and can be
        # kept for reference or manually cleaned up periodically.
        
        # Count matches that will be deleted (for reporting)
        matches_before = len(db.get_top_matches(limit=1000))
        
        # Delete the job from database (this now cascades to matches)
        success = db.delete_job(job_id)
        
        if success:
            # Count how many matches were deleted
            matches_after = len(db.get_top_matches(limit=1000))
            matches_deleted = matches_before - matches_after
            
            message = 'Job deleted successfully'
            if matches_deleted > 0:
                message += f' (and {matches_deleted} associated matches)'
                
            return jsonify({'success': True, 'message': message, 'matches_deleted': matches_deleted})
        else:
            return jsonify({'success': False, 'error': 'Failed to delete job'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/update_resume_skills/<resume_id>', methods=['PUT'])
def update_resume_skills(resume_id):
    """Update skills for a resume"""
    try:
        from graph.database import graphDB
        db = graphDB()
        
        data = request.json
        skills = data.get('skills', [])
        
        # Update the resume's skills
        success = db.update_resume_skills(resume_id, skills)
        
        if success:
            return jsonify({'success': True, 'message': 'Skills updated successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to update skills'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear_data', methods=['POST'])
def clear_data():
    """Clear jobs and matches from database and files"""
    try:
        data = request.json
        clear_jobs = data.get('clear_jobs', False)
        clear_matches = data.get('clear_matches', False)
        clear_resumes = data.get('clear_resumes', False)  # Optional, default false for safety
        
        from graph.database import graphDB
        import shutil
        import glob
        
        db = graphDB()
        cleared_items = []
        
        # Clear matches from database
        if clear_matches:
            try:
                # Clear matches table
                db.drop_table('matches')
                db._init_tables()  # Recreate empty table
                cleared_items.append('matches from database')
                
                # Clear match output files
                match_files = glob.glob('data/pipeline_output/matches/*.json')
                for file in match_files:
                    os.remove(file)
                if match_files:
                    cleared_items.append(f'{len(match_files)} match output files')
            except Exception as e:
                print(f"Error clearing matches: {e}")
        
        # Clear jobs from database and files
        if clear_jobs:
            try:
                # Clear jobs table
                db.drop_table('jobs')
                db._init_tables()  # Recreate empty table
                cleared_items.append('jobs from database')
                
                # Clear job files
                job_files = [
                    'data/pipeline_output/jobs/bulk_latest.json',
                    'data/pipeline_output/jobs/processed_jobs.json'
                ]
                # Also clear bulk job files with timestamps
                job_files.extend(glob.glob('data/pipeline_output/jobs/bulk_jobs_*.json'))
                
                deleted_count = 0
                for file in job_files:
                    if os.path.exists(file):
                        os.remove(file)
                        deleted_count += 1
                
                if deleted_count > 0:
                    cleared_items.append(f'{deleted_count} job files')
            except Exception as e:
                print(f"Error clearing jobs: {e}")
        
        # Clear resumes (optional - usually want to keep these)
        if clear_resumes:
            try:
                # Clear resumes table
                db.drop_table('resumes')
                db._init_tables()  # Recreate empty table
                cleared_items.append('resumes from database')
                
                # Clear uploaded resume files
                upload_files = glob.glob('data/uploads/*')
                for file in upload_files:
                    os.remove(file)
                if upload_files:
                    cleared_items.append(f'{len(upload_files)} uploaded resume files')
            except Exception as e:
                print(f"Error clearing resumes: {e}")
        
        if cleared_items:
            message = f"Successfully cleared: {', '.join(cleared_items)}"
        else:
            message = "No data was cleared"
        
        return jsonify({
            'success': True,
            'message': message,
            'cleared': cleared_items
        })
        
    except Exception as e:
        print(f"Error clearing data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export_matches')
def export_matches():
    """Export matches to CSV"""
    try:
        from graph.database import graphDB
        db = graphDB()
        matches = db.get_top_matches(limit=100)
        
        # Get all jobs and resumes for lookup
        all_jobs_list = db.get_all_jobs()
        all_resumes_list = db.get_all_resumes()
        
        # Create lookup dictionaries - handle both job_id formats
        all_jobs = {job['job_id']: job for job in all_jobs_list}
        # Also add lookup by job_index for backward compatibility
        for idx, job in enumerate(all_jobs_list):
            all_jobs[str(idx)] = job  # Add by index
            all_jobs[str(job.get('job_index', idx))] = job  # Add by job_index if exists
        
        all_resumes = {resume['resume_id']: resume for resume in all_resumes_list}
        
        # Create CSV content
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Resume Name', 'Job Title', 'Company', 'Location',
            'Overall Score', 'Skills Score', 'Experience Score',
            'Semantic Score', 'LLM Score', 'Matched At'
        ])
        
        # Write data
        for match in matches:
            # Get job and resume details
            job_id = match.get('job_id', '')
            resume_id = match.get('resume_id', '')
            
            # Look up job details
            job = all_jobs.get(job_id, {})
            job_title = job.get('title', 'Unknown Position')
            company = job.get('company', 'Unknown Company')
            location = job.get('location', 'Unknown Location')
            
            # Look up resume details
            resume = all_resumes.get(resume_id, {})
            resume_name = resume.get('name', 'Unknown')
            
            # If name is still unknown, try to get from filename
            if resume_name == 'Unknown' and resume.get('filename'):
                resume_name = resume.get('filename', '').replace('.pdf', '').replace('.docx', '').replace('_', ' ')
            
            writer.writerow([
                resume_name,
                job_title,
                company,
                location,
                round(match.get('overall_score', 0) * 100),
                round(match.get('skills_score', 0) * 100),
                round(match.get('experience_score', 0) * 100),
                round(match.get('semantic_score', 0) * 100),
                round(match.get('llm_score', 0) * 100),
                match.get('match_timestamp', '')
            ])
        
        # Create response
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'matches_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@socketio.on('connect')
def handle_connect(auth):
    """Handle client connection"""
    try:
        from flask import request
        client_id = request.sid
        connected_clients.add(client_id)
        emit('connected', {'message': 'Connected to server'})
        print(f'Client connected: {client_id} (Total: {len(connected_clients)})')
        
        # Start background progress emitter on first connection
        if not hasattr(handle_connect, 'emitter_started'):
            handle_connect.emitter_started = True
            socketio.start_background_task(background_progress_emitter)
            print("Progress emitter background task started")
    except Exception as e:
        print(f"Error in handle_connect: {e}")
        return False  # Reject connection on error

@socketio.on('disconnect')
def handle_disconnect(auth=None):
    """Handle client disconnection"""
    try:
        from flask import request
        client_id = request.sid
        connected_clients.discard(client_id)
        print(f'Client disconnected: {client_id} (Remaining: {len(connected_clients)})')
    except Exception as e:
        print(f"Error in handle_disconnect: {e}")

@socketio.on_error_default
def default_error_handler(e):
    """Handle SocketIO errors"""
    print(f"SocketIO error: {e}")
    return False  # Don't propagate the error

@socketio.on('ping')
def handle_ping():
    """Handle ping from client for keepalive"""
    emit('pong', {'timestamp': datetime.now().isoformat()})

# Background task for processing progress updates using SocketIO's task system
def background_progress_emitter():
    """Background task that processes the progress queue"""
    while True:
        try:
            # Only process if we have connected clients
            if connected_clients and not progress_queue.empty():
                try:
                    update = progress_queue.get_nowait()
                    # Emit to all connected clients
                    try:
                        socketio.emit(update['type'], update['data'], namespace='/')
                        print(f"Emitted {update['type']} event: {update['data'].get('message', '')[:50]}")
                    except Exception as emit_error:
                        # Silently skip werkzeug assertion errors
                        error_str = str(emit_error)
                        if "write() before start_response" not in error_str and "AssertionError" not in error_str:
                            print(f"Could not emit update: {emit_error}")
                except queue.Empty:
                    pass
                except Exception as e:
                    if "AssertionError" not in str(e):
                        print(f"Error processing update: {e}")
            
            socketio.sleep(0.1)  # Use socketio.sleep for better integration
        except Exception as e:
            # Don't log assertion errors from werkzeug
            error_str = str(e)
            if "write() before start_response" not in error_str and "AssertionError" not in error_str:
                print(f"Error in progress emitter: {e}")
            socketio.sleep(1)  # Back off on error

if __name__ == '__main__':
    print("Starting Flask app on port 5000...")
    
    # Start LLM server before starting Flask
    start_llm_server()
    
    # Disable debug mode to prevent auto-reloading when pipeline imports trigger
    socketio.run(app, debug=False, port=5000, use_reloader=False, host='0.0.0.0', allow_unsafe_werkzeug=True)
