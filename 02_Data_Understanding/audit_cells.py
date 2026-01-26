import json
import sys

def audit_notebook(notebook_path, output_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    report_lines = []
    report_lines.append(f"# Detailed Cell-by-Cell Audit Report: {notebook_path.split('/')[-1]}")
    report_lines.append("")
    report_lines.append("| Cell # | Type | Exec # | Status | Content Summary | Issues/Notes |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    cells = nb.get('cells', [])
    
    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type', 'unknown')
        source = "".join(cell.get('source', [])).strip()
        summary = source.split('\n')[0][:50] + "..." if source else "(Empty)"
        summary = summary.replace("|", "\|").replace("`", "") # Sanitize for table
        
        exec_count = "N/A"
        status = "OK"
        issues = []

        if cell_type == 'code':
            exec_count = cell.get('execution_count')
            if exec_count is None:
                exec_count = "None"
                # Check if it has source. If source exists but no exec count, might be unrun.
                if source:
                    status = "Not Run"
            
            outputs = cell.get('outputs', [])
            for output in outputs:
                if output.get('output_type') == 'error':
                    status = "ERROR"
                    ename = output.get('ename', 'Error')
                    evalue = output.get('evalue', '')
                    issues.append(f"**Error:** {ename}: {evalue}")
                elif output.get('name') == 'stderr':
                    # Sometimes stderr is just warnings, but worth noting
                    issues.append("Has stderr output")

        elif cell_type == 'markdown':
            # Basic markdown checks? 
            # Maybe check for empty markdown
            if not source:
                status = "Empty"
                issues.append("Empty markdown cell")

        issue_text = "<br>".join(issues) if issues else "-"
        
        # Format the row
        row = f"| {i+1} | {cell_type} | {exec_count} | {status} | `{summary}` | {issue_text} |"
        report_lines.append(row)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"Audit report generated at: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python audit_cells.py <notebook_path> <output_report_path>")
    else:
        audit_notebook(sys.argv[1], sys.argv[2])
