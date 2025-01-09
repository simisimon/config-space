// Force-Directed Graph Visualization using d3.js

function updateLabels(labelSelection, nodeData) {
    labelSelection.text(d => {
        if (d.type === "option") {
            return d.id.split(":")[1]; // Only show the option name
        }
        return d.id; // For concepts and artifacts, show the id directly
    });
}

// Set up the dimensions and margins for the SVG container
let width = window.innerWidth;
let height = window.innerHeight;

// Create an SVG element and append it to the body
const svg = d3.select("body")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

// Add zoom behavior
const zoom = d3.zoom()
    .scaleExtent([0.5, 5]) // Set min and max zoom levels
    .on("zoom", (event) => {
        svg.select("g").attr("transform", event.transform); // Apply zoom
    });

// Apply zoom behavior to the SVG
svg.call(zoom);

// Create a container group for all elements
const container = svg.append("g");

// Load the graph data from the JSON file
d3.json("../data/test_data/test_graph_data_final.json").then((graph) => {
    // Filter nodes and links to include only concepts and artifacts
    const filteredLinks = graph.links.filter(d =>
        graph.nodes.some(node => node.id === d.source) &&
        graph.nodes.some(node => node.id === d.target)
    );

    // Collect IDs of connected nodes
    const connectedNodeIds = new Set();
    filteredLinks.forEach(link => {
        connectedNodeIds.add(link.source);
        connectedNodeIds.add(link.target);
    });

    // Filter nodes to remove unlinked ones
    const filteredNodes = graph.nodes.filter(node => connectedNodeIds.has(node.id));

    // Debug removed nodes
    const removedNodes = graph.nodes.filter(node => !connectedNodeIds.has(node.id));
    console.log("Removed Nodes (Unlinked):", removedNodes);

    // Create linkForce
    const linkForce = d3.forceLink(filteredLinks)
        .id(d => d.id)
        .distance(d => { return 50; })  // Default distance for other links
        .strength(d => { return 0.3; }); // Adjust strength of links


    // Use filtered nodes and links for the simulation
    const simulation = d3.forceSimulation(filteredNodes)
        .force("link", linkForce)
        .force("charge", d3.forceManyBody().strength(-20))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(5));

    // Add links to the container
    const link = container.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(filteredLinks)
        .enter().append("line")
        .attr("stroke-width", 1.5)
        .attr("stroke", "#aaa");

    // Add nodes to the container
    const node = container.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(filteredNodes)
        .enter().append("circle")
        .attr("r", 8)
        .attr("fill", d => {
            if (d.type === 'concept') return "#1f77b4"; // Blue for concepts
            if (d.type === 'artifact') return "#ff7f0e"; // Orange for artifacts
            return "#2ca02c"; // Green for options
        })
        .call(d3.drag()
            .on("start", (event, d) => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on("drag", (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on("end", (event, d) => {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }));

    // Add labels to nodes
    const label = container.append("g")
        .attr("class", "labels")
        .selectAll("text")
        .data(filteredNodes)
        .enter().append("text")
        .attr("dy", -10)
        .attr("text-anchor", "middle")
        .text(d => d.id) // Or use `updateLabels` for more complex logic
        .style("font-size", "10px")
        .style("fill", "#555")
        .style("visibility", "hidden"); // Initially hidden

    // Show labels on mouseover
    node.on("mouseover", (event, d) => {
        label.filter(l => l.id === d.id)
            .style("visibility", "visible"); // Show label
        })
        .on("mouseout", (event, d) => {
            label.filter(l => l.id === d.id)
                .style("visibility", "hidden"); // Hide label
        });
    
    
    // Update the positions of nodes and links during the simulation
    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

        label
            .attr("x", d => d.x)
            .attr("y", d => d.y);
        
        // Dynamically update labels
        updateLabels(label, graph.nodes);
    });

    // Add a tooltip to display node information
    node.append("title")
        .text(d => `${d.type}: ${d.id}`);

}).catch(error => {
    console.error("Error loading the graph data: ", error);
});

// Add a reset button
const resetButton = d3.select("body").append("button")
    .attr("id", "reset-zoom")
    .style("position", "absolute")
    .style("top", "10px")
    .style("left", "10px")
    .style("z-index", 10)
    .text("Reset Zoom");

resetButton.on("click", () => {
    svg.transition()
        .duration(750)
        .call(zoom.transform, d3.zoomIdentity); // Reset to initial zoom
});


// Update the size of the SVG on window resize
window.addEventListener("resize", () => {
    width = window.innerWidth;
    height = window.innerHeight;

    svg.attr("width", width).attr("height", height);
});

