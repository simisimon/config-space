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

// Color scale for options
const optionColorScale = d3.scaleSequential()
    .domain([0.5, 10]) // Shift domain to start at 0.5
    .interpolator(d3.interpolateGreens)
    .clamp(true);

// Create a container group for all elements
const container = svg.append("g");

// Variable to track the state of the checkbox
let showConceptLinks = true;

// Function to load and render the graph data
function loadGraphData(fileName) {
    const filePath = `/data/test_data/graph_data/${fileName}`;
    d3.json(filePath).then((graph) => {
        // Clear previous graph
        container.selectAll("*").remove();

        // Filter nodes and links to include only concepts and artifacts
        const filteredNodes = graph.nodes.filter(d => d.type === 'concept' || d.type === 'artifact' || d.type === 'option');
        
        
        // const filteredLinks = graph.links.filter(d =>
        //     (d.type !== 'concept-concept') &&
        //     filteredNodes.some(node => node.id === d.source) &&
        //     filteredNodes.some(node => node.id === d.target));

        // Filter links excluding concept-to-concept links for simulation
        const filteredLinks = graph.links.filter(d =>
            (d.type !== 'concept-concept') &&
            filteredNodes.some(node => node.id === d.source) &&
            filteredNodes.some(node => node.id === d.target)
        );

        const conceptLinks = graph.links.filter(d => 
            (d.type == 'concept-concept') &&
            (d.weight > 0.5) &&
            filteredNodes.some(node => node.id === d.source) &&
            filteredNodes.some(node => node.id === d.target)
        );
        
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

        // Add tooltip
        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "tooltip")
            .style("position", "absolute")
            .style("background-color", "white")
            .style("border", "1px solid #ccc")
            .style("border-radius", "5px")
            .style("padding", "10px")
            .style("font-size", "12px")
            .style("pointer-events", "none")
            .style("visibility", "hidden");

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
                if (d.type === 'option') {
                    if (d.changed_internally == 0) return "#c7e9c0"
                    if (d.changed_internally > 0 && d.changed_internally <= 3) return "#41ab5d"
                    else return "#005a32"
                }
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

        const label = container.append("g")
            .attr("class", "labels")
            .selectAll("text")
            .data(filteredNodes)
            .enter().append("text")
            .filter(d => d.type === 'concept' || d.type === 'artifact') // Only for concepts and artifacts
            .attr("dy", -10) // Position above the node
            .attr("text-anchor", "middle")
            .text(d => d.id)
            .style("font-size", "12px")
            .style("fill", "#333")
            .style("fill", d => d.type === 'concept' ? "#1f77b4" : "#333") // Match color for concepts
            .style("visibility", d => d.type === 'concept' ? "visible" : "hidden"); // Always visible for concepts

        // Add concept-to-concept links without force
        const conceptLinkGroup = container.append("g")
            .attr("class", "concept-links");

        const conceptLink = conceptLinkGroup.selectAll("line")
            .data(conceptLinks)
            .enter().append("line")
            .attr("stroke-width", 1.5)
            .attr("stroke", "#00f") // Blue color for concept links
            .attr("x1", d => filteredNodes.find(n => n.id === d.source).x)
            .attr("y1", d => filteredNodes.find(n => n.id === d.source).y)
            .attr("x2", d => filteredNodes.find(n => n.id === d.target).x)
            .attr("y2", d => filteredNodes.find(n => n.id === d.target).y);

        // Add labels for concept-to-concept links
        const conceptLinkLabel = conceptLinkGroup.selectAll("text")
            .data(conceptLinks)
            .enter().append("text")
            .attr("text-anchor", "middle")
            .attr("dy", -5) // Slightly above the link
            .style("font-size", "10px")
            .style("fill", "#00f") // Match link color
            .text(d => { return `${d.count} (${d.weight})` })
            .attr("x", d => {
                const sourceNode = filteredNodes.find(n => n.id === d.source);
                const targetNode = filteredNodes.find(n => n.id === d.target);
                return (sourceNode.x + targetNode.x) / 2; // Midpoint x
            })
            .attr("y", d => {
                const sourceNode = filteredNodes.find(n => n.id === d.source);
                const targetNode = filteredNodes.find(n => n.id === d.target);
                return (sourceNode.y + targetNode.y) / 2; // Midpoint y
            });
        
        
        node.on("mouseover", (event, d) => {
            if (d.type === 'artifact') {
                // Show label for the hovered node
                label.filter(l => l.id === d.id)
                    .style("visibility", "visible");
            } else if (d.type === 'option') {
                // Show tooltip for option nodes
                tooltip.style("visibility", "visible")
                    .html(`
                        <strong>Name:</strong> ${d.id.split(":")[1]}<br>
                        <strong>Values:</strong> ${d.values}<br>
                        <strong>Changed Internally:</strong> ${d.changed_internally}<br>
                        <strong>Changed Globally:</strong> ${d.changed_globally}
                    `);
            }
        })
            .on("mousemove", (event) => {
                // Update tooltip position for option nodes
                tooltip.style("top", `${event.pageY + 10}px`)
                    .style("left", `${event.pageX + 10}px`);
            })
            .on("mouseout", (event, d) => {
                if (d.type === 'artifact') {
                    // Hide label when mouse leaves the node
                    label.filter(l => l.id === d.id)
                        .style("visibility", "hidden");
                } else if (d.type === 'option') {
                    // Hide tooltip for option nodes
                    tooltip.style("visibility", "hidden");
                }
            });

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
            
            // Update concept-to-concept links manually
            conceptLink
                .attr("x1", d => filteredNodes.find(n => n.id === d.source).x)
                .attr("y1", d => filteredNodes.find(n => n.id === d.source).y)
                .attr("x2", d => filteredNodes.find(n => n.id === d.target).x)
                .attr("y2", d => filteredNodes.find(n => n.id === d.target).y);

            conceptLinkLabel
                .attr("x", d => {
                    const sourceNode = filteredNodes.find(n => n.id === d.source);
                    const targetNode = filteredNodes.find(n => n.id === d.target);
                    return (sourceNode.x + targetNode.x) / 2; // Midpoint x
                })
                .attr("y", d => {
                    const sourceNode = filteredNodes.find(n => n.id === d.source);
                    const targetNode = filteredNodes.find(n => n.id === d.target);
                    return (sourceNode.y + targetNode.y) / 2; // Midpoint y
                });

            // Dynamically update labels
            updateLabels(label, graph.nodes);
        });

        // Toggle concept link visibility
        document.getElementById('toggle-concept-links-checkbox').addEventListener('change', (event) => {
            const isChecked = event.target.checked;
            conceptLink.style("visibility", isChecked ? "visible" : "hidden");
            conceptLinkLabel.style("visibility", isChecked ? "visible" : "hidden");
        });

        // node.each(function (d) {
        //     console.log(d.id, this.getAttribute("stroke"), this.getAttribute("stroke-width"));
        // });

    }).catch(error => {
        console.error("Error loading the graph data: ", error);
    });
}

// Add event listener to the dropdown menu
document.getElementById('data-file-selector').addEventListener('change', function () {
    const selectedFile = this.value;
    console.log("Selected file:", selectedFile); // Debugging line
    loadGraphData(selectedFile); // Load new graph data
});

// Load the initial graph data
loadGraphData(document.getElementById('data-file-selector').value);

// Add a reset button
// const resetButton = d3.select("body").append("button")
//     .attr("id", "reset-zoom")
//     .style("position", "absolute")
//     .style("top", "10px")
//     .style("left", "10px")
//     .style("z-index", 10)
//     .text("Reset Zoom");

// resetButton.on("click", () => {
//      svg.transition()
//          .duration(750)
//          .call(zoom.transform, d3.zoomIdentity); // Reset to initial zoom
// });

const legendData = [
    { type: "Technology", color: "#1f77b4" },
    { type: "Configuration File", color: "#ff7f0e" },
    { type: "Configuration Option", color: "#2ca02c" }
];

// Add legend for discrete items
const legend = svg.append("g")
    .attr("class", "legend")
    .attr("transform", "translate(20, 80)");

legend.selectAll("legend-item")
    .data(legendData)
    .enter().append("g")
    .attr("class", "legend-item")
    .attr("transform", (d, i) => `translate(0, ${i * 20})`)
    .each(function (d) {
        const item = d3.select(this);

        // Add color circle
        item.append("circle")
            .attr("cx", 0)
            .attr("cy", 0)
            .attr("r", 6)
            .attr("fill", d.color);

        // Add text label
        item.append("text")
            .attr("x", 15)
            .attr("y", 5)
            .style("font-size", "14px")
            .style("fill", "#000")
            .text(d.type);
    });

// Add the gradient legend
const gradientLegend = svg.append("g")
    .attr("class", "gradient-legend")
    .attr("transform", "translate(20, 180)");

gradientLegend.append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", 200)
    .attr("height", 10)
    .style("fill", "url(#option-gradient)");

gradientLegend.append("text")
    .attr("x", 0)
    .attr("y", -5)
    .style("font-size", "12px")
    .style("fill", "#000")
    .text("Not Changed");

gradientLegend.append("text")
    .attr("x", 200)
    .attr("y", -5)
    .attr("text-anchor", "end")
    .style("font-size", "12px")
    .style("fill", "#000")
    .text("Frequently Changed");

gradientLegend.append("text")
    .attr("x", 0)
    .attr("y", -20)
    .style("font-size", "14px")
    .style("font-weight", "bold")
    .style("fill", "#000")
    .text("Changed Internally");

const defs = svg.append("defs");

const linearGradient = defs.append("linearGradient")
    .attr("id", "option-gradient")
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "100%")
    .attr("y2", "0%");

linearGradient.append("stop")
   .attr("offset", "0%")
   .attr("stop-color", "#c7e9c0");

linearGradient.append("stop")
   .attr("offset", "50%")
   .attr("stop-color", "#41ab5d");

linearGradient.append("stop")
   .attr("offset", "100%")
    .attr("stop-color", "#005a32");


// // Update the size of the SVG on window resize
window.addEventListener("resize", () => {
    width = window.innerWidth;
    height = window.innerHeight;

    svg.attr("width", width).attr("height", height);
});

