import os
import re
import requests
from typing import List, Dict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 180
BASE_OUTPUT_DIR = "projects"
work_dir = BASE_OUTPUT_DIR
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

print(f"🦙 Ollama model: {OLLAMA_MODEL}")
print(f"📡 Endpoint: {OLLAMA_BASE_URL}")
print(f"🚀 GPU: ENABLED")
print(f"⚡ PARALLEL EXECUTION: ON\n")

class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        
    def chat(self, messages: List[Dict], stream: bool = False) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": stream, "options": {"temperature": 0.7, "num_ctx": 8192}}
        try:
            response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
            response.raise_for_status()
            return response.json()["message"]["content"] if not stream else response.text
        except Exception as e:
            return f"Error: {str(e)}"

ollama_client = OllamaClient(OLLAMA_BASE_URL, OLLAMA_MODEL)

@dataclass
class ProjectState:
    project_requirements: str = ""
    num_developers: int = 0
    developers: List[str] = field(default_factory=list)
    project_plan: str = ""
    development_results: Dict[str, str] = field(default_factory=dict)
    qa_results: List[str] = field(default_factory=list)
    qa_approved: bool = False
    extracted_files: Dict[str, str] = field(default_factory=dict)

project_state = ProjectState()

def extract_code_files(text: str, output_dir: str) -> Dict[str, str]:
    files = {}
    pattern1 = r'####\s+`?([^\n:`]+\.(html|js|jsx|json|md|txt|py|cpp|c|java|go|rs|php|rb|css|tsx|ts))`?:\s*```(\w+)?\n(.*?)```'
    for match in re.finditer(pattern1, text, re.DOTALL):
        filename = match.group(1).strip()
        code = match.group(4).strip()
        if len(code) > 10:
            files[filename] = code
    pattern2 = r'completed `([^`]+\.(html|js|jsx|json|md|txt|py|cpp|c|java|go|rs|php|rb|css|tsx|ts))`[:\s]+```(\w+)?\n(.*?)```'
    for match in re.finditer(pattern2, text, re.DOTALL):
        filename = match.group(1).strip()
        code = match.group(4).strip()
        if filename not in files and len(code) > 10:
            files[filename] = code
    pattern3 = r'Here is the completed `([^`]+\.(html|js|jsx|json|md|txt|py|cpp|c|java|go|rs|php|rb|css|tsx|ts))`:\s*```(\w+)?\n(.*?)```'
    for match in re.finditer(pattern3, text, re.DOTALL):
        filename = match.group(1).strip()
        code = match.group(4).strip()
        if filename not in files and len(code) > 10:
            files[filename] = code
    saved = 0
    for filepath, code in files.items():
        full_path = os.path.join(output_dir, filepath)
        dirname = os.path.dirname(full_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"   ✅ {filepath} ({len(code)} bytes)")
            saved += 1
        except Exception as e:
            print(f"   ❌ {filepath}: {e}")
    if saved > 0:
        print(f"\n💾 Saved {saved} files to {output_dir}/")
    return files

@dataclass
class Agent:
    name: str
    role: str
    system_prompt: str
    
    def think(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]
        response = ollama_client.chat(messages)
        print(f"✅ {self.name}: Task completed")
        return response

def create_orchestrator() -> Agent:
    return Agent(
        name="Orchestrator",
        role="System Coordinator",
        system_prompt="You are the Orchestrator. Analyze project requirements and decide team size (1-4 developers). For simple projects: 1 dev. Medium: 2 devs. Complex/Full-stack: 3-4 devs. Be concise."
    )

def create_hr_agent() -> Agent:
    return Agent(
        name="HR",
        role="Human Resources",
        system_prompt="You are HR. Confirm hiring/termination. Be brief: 'Hired X developers' or 'Terminated team'."
    )

def create_manager() -> Agent:
    return Agent(
        name="Manager",
        role="Project Manager",
        system_prompt="You are the Manager. Create detailed technical plans with file structure, tech stack, and task assignments for each developer. Be specific about what each developer builds."
    )

def create_developer(dev_id: int) -> Agent:
    return Agent(
        name=f"Developer_{dev_id}",
        role=f"Software Developer {dev_id}",
        system_prompt=f"""You are Developer {dev_id}. Write COMPLETE, PRODUCTION-READY code.

CRITICAL FORMATTING RULES:
1. Use this EXACT format for EVERY file:
   #### `filename.ext`:
   ```language
   [complete code here]
   ```

2. Include ALL necessary files (HTML, CSS, JS, config files, README)
3. Write COMPLETE code - no placeholders, no comments like "add more here"
4. Make it WORK out of the box

Example:
#### `index.html`:
```html
<!DOCTYPE html>
<html>
<head><title>App</title></head>
<body><h1>Hello</h1></body>
</html>
```

#### `app.js`:
```javascript
console.log('Working!');
```

Now complete YOUR assigned task with FULL, WORKING CODE."""
    )

def create_qa_agent(qa_id: int, focus: str) -> Agent:
    return Agent(
        name=f"QA_{qa_id}",
        role=f"Quality Assurance ({focus})",
        system_prompt=f"You are QA Agent {qa_id} focusing on {focus}. Review code. Be critical. End with 'APPROVE' or 'REJECT: [reason]'. Be concise."
    )

def analyze_project_complexity(project_request: str) -> int:
    request_lower = project_request.lower()
    fullstack_keywords = ['full stack', 'fullstack', 'full-stack', 'ecommerce', 'e-commerce', 'authentication', 'auth', 'database', 'backend', 'frontend', 'api', 'rest', 'graphql', 'crud', 'user management']
    complex_keywords = ['dashboard', 'admin panel', 'chat', 'real-time', 'websocket', 'payment', 'shopping cart', 'social media', 'blog platform']
    medium_keywords = ['multi-page', 'form validation', 'data visualization', 'interactive', 'dynamic', 'react', 'vue', 'angular', 'component']
    fullstack_count = sum(1 for kw in fullstack_keywords if kw in request_lower)
    complex_count = sum(1 for kw in complex_keywords if kw in request_lower)
    medium_count = sum(1 for kw in medium_keywords if kw in request_lower)
    if fullstack_count >= 2 or complex_count >= 2:
        return 4
    elif fullstack_count >= 1 or complex_count >= 1:
        return 3
    elif medium_count >= 2 or len(project_request.split()) > 20:
        return 2
    else:
        return 1

def run_multi_agent_system(project_request: str):
    global work_dir
    project_name = re.sub(r'[^\w\s-]', '', project_request[:50]).strip().replace(' ', '_')
    if not project_name:
        project_name = "project"
    work_dir = os.path.join(BASE_OUTPUT_DIR, project_name)
    os.makedirs(work_dir, exist_ok=True)
    print("\n" + "="*80)
    print("🚀 MULTI-AGENT ORCHESTRATION SYSTEM STARTING")
    print("="*80 + "\n")
    print(f"📁 Project Folder: {work_dir}/\n")
    project_state.project_requirements = project_request
    print("\n📋 [STEP 1] PROJECT ANALYSIS")
    num_devs = analyze_project_complexity(project_request)
    project_state.num_developers = num_devs
    print(f"✅ Decision: {project_state.num_developers} developers (Smart Analysis)")
    print(f"\n👥 [STEP 2] HIRING")
    hr_agent = create_hr_agent()
    hr_agent.think(f"Confirm hiring {project_state.num_developers} developers.")
    project_state.developers = [f"Developer_{i+1}" for i in range(project_state.num_developers)]
    print(f"✅ Team: {', '.join(project_state.developers)}")
    print(f"\n📝 [STEP 3] PLANNING")
    manager = create_manager()
    project_plan = manager.think(f"Create plan:\n{project_request}\nTeam: {', '.join(project_state.developers)}")
    project_state.project_plan = project_plan
    print(f"✅ Plan created")
    print(f"\n💻 [STEP 4] DEVELOPMENT (PARALLEL EXECUTION)")
    developers = [create_developer(i+1) for i in range(project_state.num_developers)]
    def dev_work(dev):
        result = dev.think(f"Plan:\n{project_plan[:800]}\n\nProject:\n{project_request}\n\nComplete YOUR assigned task. Use format: #### `filename.ext`: for ALL code files!")
        return (dev.name, result)
    with ThreadPoolExecutor(max_workers=project_state.num_developers) as executor:
        futures = [executor.submit(dev_work, dev) for dev in developers]
        for future in as_completed(futures):
            name, result = future.result()
            project_state.development_results[name] = result
    print(f"✅ Development complete ({project_state.num_developers} developers worked in parallel)")
    print(f"\n🔧 [STEP 5] INTEGRATION")
    integration = manager.think(f"Integrate:\nProject: {project_request}\n\nOutputs:\n" + "\n".join([f"{n}:\n{r[:300]}" for n, r in project_state.development_results.items()]))
    print(f"✅ Integrated")
    print(f"\n🔍 [STEP 6] QA")
    qa_agents = [create_qa_agent(i+1, ["Functional", "Code Quality", "Integration"][i]) for i in range(3)]
    approve_count = 0
    for qa in qa_agents:
        result = qa.think(f"Review:\nRequest: {project_request}\n\nProject:\n{integration[:800]}")
        project_state.qa_results.append(result)
        if "APPROVE" in result.upper() and "REJECT" not in result.upper():
            approve_count += 1
    project_state.qa_approved = approve_count >= 2
    print(f"{'✅ APPROVED' if project_state.qa_approved else '⚠️ NEEDS REVIEW'} QA: {approve_count}/3")
    print(f"\n👋 [STEP 7] CLEANUP")
    hr_agent.think(f"Terminate {len(project_state.developers)} developers.")
    project_state.developers.clear()
    print(f"✅ Terminated")
    print(f"\n📦 [STEP 8] DELIVERY & FILE EXTRACTION")
    summary = create_orchestrator().think(f"Summary:\nProject: {project_request}\nQA: {'APPROVED' if project_state.qa_approved else 'REVISION'}\nDevs: {project_state.num_developers}")
    print(f"\n📝 Extracting code files from integration...")
    extract_code_files(integration, work_dir)
    print(f"\n📝 Extracting code files from individual developers...")
    for dev_name, dev_result in project_state.development_results.items():
        extract_code_files(dev_result, work_dir)
    report = os.path.join(work_dir, "project_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("PROJECT REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Request: {project_request}\n\n")
        f.write(f"Team Size: {project_state.num_developers} developers\n\n")
        f.write("="*80 + "\n")
        f.write("PLAN:\n")
        f.write("="*80 + "\n")
        f.write(project_plan + "\n\n")
        f.write("="*80 + "\n")
        f.write("DEVELOPMENT:\n")
        f.write("="*80 + "\n")
        for name, result in project_state.development_results.items():
            f.write(f"\n{name}:\n{'-'*40}\n{result}\n\n")
        f.write("="*80 + "\n")
        f.write("INTEGRATION:\n")
        f.write("="*80 + "\n")
        f.write(integration + "\n\n")
        f.write("="*80 + "\n")
        f.write("QA RESULTS:\n")
        f.write("="*80 + "\n")
        for i, result in enumerate(project_state.qa_results, 1):
            f.write(f"\nQA_{i}:\n{result}\n\n")
        f.write(f"\nFinal Status: {'APPROVED ✅' if project_state.qa_approved else 'NEEDS REVISION ⚠️'}\n")
    all_files = []
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), work_dir)
            all_files.append(rel_path)
    print("\n" + "="*80)
    print("✅ SYSTEM COMPLETED")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Project: {project_request[:60]}...")
    print(f"   Developers: {project_state.num_developers}")
    print(f"   QA: {'APPROVED ✅' if project_state.qa_approved else 'REVISION ⚠️'}")
    print(f"   Files Created: {len(all_files)}")
    print(f"   Output Directory: {work_dir}/\n")
    print("📁 Created Files:")
    for file in sorted(all_files):
        file_path = os.path.join(work_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file} ({size} bytes)")
    print()

if __name__ == "__main__":
    project_request = input("🎯 Enter project request: ").strip()
    if not project_request:
        print("❌ No project request provided!")
    else:
        run_multi_agent_system(project_request)
