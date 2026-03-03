
import csv
import json
import os

def create_visualization():
    csv_path = "graph/artifacts_pruned/relational_prior.csv"
    output_path = "documentation/graph_structure.html"
    
    # Read the CSV
    nodes = []
    edges = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        # Headers[0] is empty, rest are node names
        node_names = headers[1:]
        
        # Create nodes list
        for i, name in enumerate(node_names):
            # Assign a group based on name (simple heuristic)
            group = "Other"
            if any(x in name.lower() for x in ["blood pressure", "heart", "respiratory", "oxygen", "temperature"]):
                group = "Vital Signs"
            elif any(x in name.lower() for x in ["glucose", "potassium", "sodium", "chloride", "creatinine", "bun", "lactate"]):
                group = "Lab Values"
            elif any(x in name.lower() for x in ["white", "platelet", "hemoglobin", "hematocrit"]):
                group = "Blood Count"

            nodes.append({"id": i, "label": name, "group": group})
            
        # Create edges list
        for i, row in enumerate(reader):
            # row[0] is the row name, same as headers[i+1]
            # row[1:] are the values
            row_name = row[0]
            values = row[1:]
            
            for j, val in enumerate(values):
                weight = float(val)
                # Filter self-loops and very weak edges for visualization clarity?
                # The user wants to see the structure. Let's show everything > 0.05 or so,
                # but distinctive weights.
                # Actually, show everything but self-loops with thin lines for weak weights.
                if i != j and weight > 0.01:
                    edges.append({
                        "from": i, 
                        "to": j, 
                        "value": weight, 
                        "title": f"{node_names[i]} -> {node_names[j]}: {weight:.2f}",
                        "color": {"opacity": min(1.0, weight)}
                    })

    # Prepare data for Vis.js
    data = {
        "nodes": nodes,
        "edges": edges
    }
    
    # HTML Template with Vis.js and Controls
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <title>MIMIC-IV Relational Prior (17 Features)</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style type="text/css">
    body {{ font-family: sans-serif; display: flex; flex-direction: column; height: 100vh; margin: 0; }}
    #controls {{ padding: 10px; background: #f0f0f0; border-bottom: 1px solid #ccc; }}
    #mynetwork {{ flex-grow: 1; width: 100%; }}
    label {{ margin-right: 10px; font-weight: bold; }}
  </style>
</head>
<body>
  <div id="controls">
    <label for="edgeFilter">Minimum Edge Weight:</label>
    <input type="range" id="edgeFilter" min="0" max="1" step="0.01" value="0.05" oninput="updateFilter()">
    <span id="edgeValue">0.05</span>
  </div>
  <div id="mynetwork"></div>
  <script type="text/javascript">
    var allNodes = {json.dumps(nodes)};
    var allEdges = {json.dumps(edges)};
    
    var nodes = new vis.DataSet(allNodes);
    var edges = new vis.DataSet(allEdges);

    var container = document.getElementById('mynetwork');
    var data = {{
      nodes: nodes,
      edges: edges
    }};
    var options = {{
      nodes: {{
        shape: 'dot',
        size: 20,
        font: {{ size: 16 }}
      }},
      edges: {{
        arrows: 'to',
        scaling: {{ min: 1, max: 10 }},
        color: {{ inherit: 'from' }},
        smooth: {{ type: 'continuous' }}
      }},
      physics: {{
        forceAtlas2Based: {{
            gravitationalConstant: -50,
            centralGravity: 0.01,
            springLength: 100,
            springConstant: 0.08
        }},
        maxVelocity: 50,
        solver: 'forceAtlas2Based',
        timestep: 0.35,
        stabilization: {{ iterations: 150 }}
      }},
      groups: {{
          "Vital Signs": {{ color: {{ background: '#FF9999', border: '#FF5555' }} }},
          "Lab Values": {{ color: {{ background: '#9999FF', border: '#5555FF' }} }},
          "Blood Count": {{ color: {{ background: '#99FF99', border: '#55AA55' }} }}
      }}
    }};
    var network = new vis.Network(container, data, options);

    function updateFilter() {{
        var val = parseFloat(document.getElementById('edgeFilter').value);
        document.getElementById('edgeValue').innerText = val;
        
        var newEdges = allEdges.filter(function(edge) {{
            return edge.value >= val;
        }});
        edges.clear();
        edges.add(newEdges);
    }}
    
    // Initial filter application
    updateFilter();
  </script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html_content)
        
    print(f"Graph visualization saved to {output_path}")

if __name__ == "__main__":
    create_visualization()
