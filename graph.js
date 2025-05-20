// Force-Directed Graph Visualization using d3.js
import { createLegend } from "./legend.js";

function updateLabels(labelSelection, nodeData) {
    labelSelection.text(d => (d.type === "option" ? d.id.split(":")[1] : d.id));
}

const dimensions = {
    width: window.innerWidth - 50,
    height: window.innerHeight - 50
};

const svg = d3.select("body")
    .append("svg")
    .attr("width", dimensions.width)
    .attr("height", dimensions.height)
    .call(
        d3.zoom()
            .scaleExtent([0.5, 5])
            .on("zoom", event => svg.select("g").attr("transform", event.transform))
    );

const container = svg.append("g");

const state = {
    showOptionLinks: false,
    topKValue: 10,
};

function filterNodes(graph) {
    return graph.nodes.filter(d => ["concept", "artifact", "option"].includes(d.type));
}

function filterLinks(graph, filteredNodes) {
    return graph.links.filter(d =>
        !["option-option"].includes(d.type) &&
        filteredNodes.some(node => node.id === d.source) &&
        filteredNodes.some(node => node.id === d.target)
    );
}

function getLinks(graph, filteredNodes, type, topKValue, commitWindow) {
    return graph.links
        .filter(d => d.type == type &&
            filteredNodes.some(node => node.id === d.source) &&
            filteredNodes.some(node => node.id === d.target) &&
            d.commit_window == commitWindow
        )
        .sort((a, b) => b.weight - a.weight)
        .slice(0, topKValue);
}

function setupSimulation(filteredNodes, filteredLinks) {
    const linkForce = d3.forceLink(filteredLinks)
        .id(d => d.id)
        .distance(d => {
            if (d.type === 'artifact-option') {
                return 100; // Increase distance for artifact-option links
            }
            return 50; // Default distance for other links
        })
        .strength(0.3);

    return d3.forceSimulation(filteredNodes)
        .force("link", linkForce)
        .force("charge", d3.forceManyBody().strength(-15))
        .force("center", d3.forceCenter(dimensions.width / 2, dimensions.height / 2))
        .force("collision", d3.forceCollide().radius(8))
        .force("x", d3.forceX(dimensions.width / 2)
            .strength(0.05) // Weak pull toward center to prevent extreme separation
        )
        .force("y", d3.forceY(dimensions.height / 2)
            .strength(0.05) // Weak pull toward center
        );
}

function setupTooltip() {
    return d3.select("body")
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
}

const tooltip = setupTooltip();

function attachTooltipListeners(optionLink) {
    optionLink
        .on("mouseover", (event, d) => {
            tooltip.style("visibility", "visible")
                .html(`
                    <strong>Link:</strong> ${d.source_option} <-> ${d.target_option} <br>
                    <strong>Changed Internal:</strong> ${ d.internal_count} (${d.internal_weight})<br>
                    <strong>Across Projects:</strong> ${d.across_projects}<br>
                `);
        })
        .on("mousemove", (event) => {
            tooltip.style("top", `${event.pageY + 10}px`)
                .style("left", `${event.pageX + 10}px`);
        })
        .on("mouseout", () => {
            tooltip.style("visibility", "hidden");
        });
}

// Global color scales for different node types
let optionColorScale;

// Function to initialize/update color scales
function updateColorScales(graph) {
    optionColorScale = d3.scaleLinear()
        .domain(d3.extent(graph.nodes.filter(d => d.type === 'option'), d => d.changed_internally))
        .range(["#c7e9c0", "#41ab5d", "#005a32"]); // Light to deep green
}

function renderGraph(graph, state, commitWindow) {
    // Clear previous graph
    container.selectAll("*").remove();

    // Filter nodes and links to include only concepts and artifacts
    const filteredNodes = filterNodes(graph);
    const filteredLinks = filterLinks(graph, filteredNodes);

    // Get links for co-changed options
    let optionLinks = getLinks(graph, filteredNodes, 'option-option', state.topKValue, commitWindow);

    const simulation = setupSimulation(filteredNodes, filteredLinks);

    updateColorScales(graph)

    const link = container.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(filteredLinks)
        .enter()
        .append("line")
        .attr("stroke-width", 1.5)
        .attr("stroke", "#aaa");

        
    const node = container.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(filteredNodes)
        .enter().append("circle")
        .attr("r", d => { return 8 })
        .attr("fill", d => {
            if (d.type === 'concept') return "#7baede"; // Blue for concepts
            if (d.type === 'artifact') return "#ff9248"; // Orange for artifacts
            if (d.type === 'option') return optionColorScale(d.changed_internally);
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

    function addLinks(linkGroup, links, visibilityState) {

        const maxInternalWeight = d3.max(links, d => d.internal_weight) || 1;
        
        const linkColorScale = d3.scaleLinear() // Fixed typo
            .domain([0, maxInternalWeight])
            .range(["#ffbaba", "#ff5252", "#a70000"]);
                
        const link = linkGroup.selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("stroke-width", d => { return 5 })
            .attr("stroke", d => { return linkColorScale(d.internal_weight) })
            .attr("x1", d => filteredNodes.find(n => n.id === d.source).x)
            .attr("y1", d => filteredNodes.find(n => n.id === d.source).y)
            .attr("x2", d => filteredNodes.find(n => n.id === d.target).x)
            .attr("y2", d => filteredNodes.find(n => n.id === d.target).y)
            .style("visibility", visibilityState ? "visible" : "hidden"); // Initially hidden

        return link;
    }

    let optionLinkGroup = container.append("g")
        .attr("class", "option-links");
    
    // Add option links without force
    let optionLink = addLinks(optionLinkGroup, optionLinks, state.showOptionLinks);

    attachTooltipListeners(optionLink)

    node.on("mouseover", (event, d) => {
        if (d.type === 'artifact') {
            label.filter(l => l.id === d.id)
                .style("visibility", "visible");
        } else if (d.type === 'option') {
            // Show tooltip for option nodes
            tooltip.style("visibility", "visible")
                .html(`
                        <strong>Name:</strong> ${d.id.split(":")[1]}<br>
                        <strong>Values:</strong> ${d.values}<br>
                        <strong>Changed Internally:</strong> ${d.changed_internally}<br>
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

        // Update option-to-option links
        optionLink
            .attr("x1", d => filteredNodes.find(n => n.id === d.source).x)
            .attr("y1", d => filteredNodes.find(n => n.id === d.source).y)
            .attr("x2", d => filteredNodes.find(n => n.id === d.target).x)
            .attr("y2", d => filteredNodes.find(n => n.id === d.target).y);

        // Dynamically update labels
        updateLabels(label, graph.nodes);
    });

    // Toggle option link visibility
    document.getElementById('toggle-option-links-checkbox').addEventListener('change', (event) => {
        state.showOptionLinks = event.target.checked;
        optionLink.style("visibility", state.showOptionLinks ? "visible" : "hidden");
    });

    // // Update the graph based on the selected top-k value
    document.getElementById('top-k-slider').addEventListener('change', (event) => {
        state.topKValue = parseInt(event.target.value);

        console.log("Top-K Value: " + state.topKValue)

        // Clear existing option links
        optionLinkGroup.selectAll("*").remove();

        // Get new top-k option links
        optionLinks = getLinks(graph, filteredNodes, 'option-option', state.topKValue, commitWindow);

        // Append new option links
        optionLink = addLinks(optionLinkGroup, optionLinks, state.showOptionLinks);

        attachTooltipListeners(optionLink)
     });

}

function loadGraphData(fileName, commitWindow) {
    console.log("File name: " + fileName)
    const filePath = `/data/graph_data/${fileName}`;
    d3.json(filePath)
        .then(graph => renderGraph(graph, state, commitWindow))
        .catch(error => console.error("Error loading graph data:", error));
}

// Load the legend
createLegend(svg);

// Load the graph data 
document.getElementById("visualize-button").addEventListener("click", () => {
    const commitWindow = document.getElementById("commit-window-size").value
    const selectedFile = document.getElementById("data-file-selector").value;
    console.log("Selected Commit Window: " + commitWindow)
    console.log("Load graph data for: " + selectedFile)
    loadGraphData(selectedFile, commitWindow);
});
