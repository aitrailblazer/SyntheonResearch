import os
import re
import hashlib
import subprocess
from datetime import date
from collections import Counter

LOGFILE = "syntheon_output.log"
RULEFILE = "syntheon_rules_glyphs.xml"
CHANGELOG = "SYNTHEON_CHANGELOG.md"

def file_md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "(not a git repo)"

def get_raw_final_verdict():
    """Extract the Final Verdict and Rule Usage sections from the log file."""
    if not os.path.exists(LOGFILE):
        return [], []
        
    with open(LOGFILE, 'r') as f:
        content = f.read()

    # Extract all content between "=== Final Verdict ===" and "Rule (and chain)"
    final_match = re.search(r'=== Final Verdict ===\n(.*?)(?=\nRule \(and chain\)|\Z)', content, re.DOTALL)
    final_section = []
    if final_match:
        final_section = final_match.group(1).strip().split('\n')
    
    # Extract rule usage statistics including all parameters
    rule_match = re.search(r'Rule \(and chain\) usage statistics \(for solved\):\n(.*?)(?=\n==|\Z)', content, re.DOTALL)
    rule_section = []
    if rule_match:
        rule_section = rule_match.group(1).strip().split('\n')
        
    return final_section, rule_section

def parse_log(logfile):
    solved_ids = set()
    rule_usage = Counter()
    total = 0
    
    # Count single rules and rule chains separately
    single_rules = Counter()
    rule_chains = Counter()

    with open(logfile) as f:
        for line in f:
            m = re.search(r"\[(train|test)\] (\w+)#(\d+) — match=(True|False) — rule_chain=([^—]+) — params=(.+)", line)
            if m:
                total += 1
                if m.group(4) == "True":
                    kind = m.group(1)
                    tid = m.group(2)
                    ex_idx = m.group(3)
                    solved_ids.add(f"{kind}:{tid}#{ex_idx}")
                    rule_chain = m.group(5).strip()
                    if rule_chain != "None":
                        rule_usage[rule_chain] += 1
                        
                        # Process individual rules and chains
                        rules = [r.strip() for r in rule_chain.split("->")]
                        if len(rules) == 1:
                            single_rules[rules[0]] += 1
                        else:
                            rule_chains[rule_chain] += 1
    
    return total, solved_ids, rule_usage, single_rules, rule_chains

def read_last_entry(changelog):
    # Parses the previous changelog entry for solved IDs and rule usage
    if not os.path.exists(changelog):
        return set(), Counter()
        
    try:
        with open(changelog) as f:
            lines = f.read().splitlines()
    except:
        return set(), Counter()
        
    # Find last entry marker
    idx = len(lines) - 1
    while idx >= 0 and not lines[idx].startswith("## "):
        idx -= 1
        
    if idx < 0:
        return set(), Counter()
        
    solved, rules = set(), Counter()
    in_solved_section = False
    
    while idx < len(lines):
        if lines[idx].startswith("- **Accuracy:**"):
            # Parse the accuracy line to extract solved count
            m = re.search(r"\((\d+) solved out of (\d+)\)", lines[idx])
            if m:
                solved_count = int(m.group(1))
                total_count = int(m.group(2))
                
        elif lines[idx].startswith("- **Rule usage:**"):
            # Start of rule usage section
            idx += 1
            # Process rules until we hit an empty line or another section
            while idx < len(lines) and (lines[idx].startswith("  - ") or lines[idx].startswith("    - ")):
                if lines[idx].startswith("  - **Rule chains:**"):
                    # Skip the rule chains header
                    idx += 1
                    continue
                    
                # Extract rule name and count
                m = re.match(r"  - (.+?): (\d+)", lines[idx])
                if m:
                    rules[m.group(1)] = int(m.group(2))
                
                # Extract rule chain and count
                m = re.match(r"    - (.+?): (\d+)", lines[idx])
                if m:
                    rules[m.group(1)] = int(m.group(2))
                    
                idx += 1
                
            continue  # Skip the increment at the end
            
        elif lines[idx].startswith("**Solved Task Changes:**"):
            in_solved_section = True
            
        elif in_solved_section and lines[idx].startswith("  + New solved:"):
            # Extract solved tasks
            solved_tasks = lines[idx][len("  + New solved:"):].strip()
            if solved_tasks:
                for task in solved_tasks.split(", "):
                    solved.add(task.strip())
                    
        elif in_solved_section and not lines[idx].strip():
            # End of solved section
            in_solved_section = False
            
        idx += 1
        
    return solved, rules

def get_log_sections(logfile):
    """Extracts both the Final Verdict and Rule Usage sections from the log file."""
    if not os.path.exists(logfile):
        print(f"Log file {logfile} not found. Using computed stats.")
        return None, None

    with open(logfile, "r") as f:
        content = f.read()

    # Extract Final Verdict section
    final_verdict = None
    final_match = re.search(r'=== Final Verdict ===\n(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    if final_match:
        final_verdict = final_match.group(0).split('\n')

    # Extract Rule Usage section
    rule_usage = None
    rule_match = re.search(r'Rule \(and chain\) usage statistics.*?:(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    if rule_match:
        rule_usage = ["Rule (and chain) usage statistics (for solved):"]
        rule_usage.extend(line.strip() for line in rule_match.group(1).split('\n') if line.strip())

    if not final_verdict:
        print("No 'Final Verdict' section found in the log file. Using computed stats.")
        return None, None

    print("'Final Verdict' section found. Extracting for changelog...")
    return final_verdict, rule_usage

def main():
    try:
        # Get raw sections
        final_verdict, rule_usage = get_raw_final_verdict()
        
        today = date.today().strftime("%Y-%m-%d")
        
        # Format the entry
        entry = []
        entry.append(f"\n## {today}: KWIC-Integrated Hybrid DSL Performance Update\n")
        
        # Add rule file hash and git commit if available
        if os.path.exists(RULEFILE):
            entry.append(f"- **Rule file hash:** `{file_md5(RULEFILE)}`")
            entry.append(f"- **Git commit:** `{git_commit_hash()}`")
        
        # Extract accuracy from final verdict
        total = correct = accuracy = None
        for line in final_verdict:
            if "Total Examples" in line:
                total = line.split(":")[-1].strip()
            elif "Correct Predictions" in line:
                correct = line.split(":")[-1].strip()
            elif "Accuracy" in line:
                accuracy = line.split(":")[-1].strip().replace("%", "").strip()
        
        if accuracy:
            entry.append(f"- **Accuracy:** **{accuracy}%** ({correct} solved out of {total})")
            
        # Add performance summary with complete solved examples and stats
        entry.append("\n### Performance Summary")
        entry.append("```")
        entry.extend(final_verdict)
        entry.append("```")
        
        # Add rule usage with complete details
        entry.append("\n### Rule Usage Statistics")
        entry.append("```")
        entry.extend(rule_usage)
        entry.append("```")

        # Write to changelog
        changelog_exists = os.path.exists(CHANGELOG)
        existing_content = ""
        if changelog_exists:
            with open(CHANGELOG, 'r') as f:
                existing_content = f.read()

        with open(CHANGELOG, 'w') as f:
            f.write("\n".join(entry))
            if existing_content:
                f.write("\n\n")
                f.write(existing_content)

        print("Changelog updated successfully!")

    except Exception as e:
        print(f"Error updating changelog: {str(e)}")
        raise

if __name__ == "__main__":
    main()
