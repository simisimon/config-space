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
    showConceptLinks: false,
    showArtifactLinks: false,
    showOptionLinks: false,
    topKValue: 10,
};

function filterNodes(graph) {
    return graph.nodes.filter(d => ["concept", "artifact", "option"].includes(d.type));
}

function filterLinks(graph, filteredNodes) {
    return graph.links.filter(d =>
        !["concept-concept", "artifact-artifact", "option-option"].includes(d.type) &&
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
    return d3.forceSimulation(filteredNodes)
        .force("link", d3.forceLink(filteredLinks).id(d => d.id).distance(50).strength(0.3))
        .force("charge", d3.forceManyBody().strength(-20))
        .force("center", d3.forceCenter(dimensions.width / 2, dimensions.height / 2))
        .force("collision", d3.forceCollide().radius(5));
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

function attachTooltipListeners(conceptLink, artifactLink, optionLink) {
    conceptLink
        .on("mouseover", (event, d) => {
            tooltip.style("visibility", "visible")
                .html(`
                    <strong>Link:</strong> ${d.source} <-> ${d.target} <br>
                    <strong>Commit Window:</strong> ${d.commit_window}<br>
                    <strong>Changed Internal:</strong> ${ d.internal_count } (${ d.internal_weight })<br>
                    <strong>Across Projects:</strong> ${d.across_projects}<br>
                    <strong>Changed Global:</strong> ${d.global_count} (${d.global_weight})
                `);
        })
        .on("mousemove", (event) => {
            tooltip.style("top", `${event.pageY + 10}px`)
                .style("left", `${event.pageX + 10}px`);
        })
        .on("mouseout", () => {
            tooltip.style("visibility", "hidden");
        });

    artifactLink
        .on("mouseover", (event, d) => {
            tooltip.style("visibility", "visible")
                .html(`
                    <strong>Link:</strong> ${d.source} <-> ${d.target} <br>
                    <strong>Commit Window:</strong> ${d.commit_window}<br>
                     <strong>Changed Internal:</strong> ${ d.internal_count} (${d.internal_weight })<br>
                     <strong>Across Projects:</strong> ${d.across_projects}<br>
                    <strong>Changed Global:</strong> ${d.global_count} (${d.global_weight})
                `);
        })
        .on("mousemove", (event) => {
            tooltip.style("top", `${event.pageY + 10}px`)
                .style("left", `${event.pageX + 10}px`);
        })
        .on("mouseout", () => {
            tooltip.style("visibility", "hidden");
        });

    optionLink
        .on("mouseover", (event, d) => {
            tooltip.style("visibility", "visible")
                .html(`
                    <strong>Link:</strong> ${d.source_option} <-> ${d.target_option} <br>
                    <strong>Changed Internal:</strong> ${ d.internal_count} (${d.internal_weight})<br>
                    <strong>Across Projects:</strong> ${d.across_projects}<br>
                    <strong>Changed Global:</strong> ${d.global_count} (${d.global_weight})
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

function renderGraph(graph, state, commitWindow) {
    // Clear previous graph
    container.selectAll("*").remove();

    // Filter nodes and links to include only concepts and artifacts
    const filteredNodes = filterNodes(graph);
    const filteredLinks = filterLinks(graph, filteredNodes);

    // Get links for different types
    let conceptLinks = getLinks(graph, filteredNodes, 'concept-concept', state.topKValue, commitWindow);
    let artifactLinks = getLinks(graph, filteredNodes, 'artifact-artifact', state.topKValue, commitWindow);
    let optionLinks = getLinks(graph, filteredNodes, 'option-option', state.topKValue, commitWindow);

    const simulation = setupSimulation(filteredNodes, filteredLinks);

    const link = container.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(filteredLinks)
        .enter()
        .append("line")
        .attr("stroke-width", 1.5)
        .attr("stroke", "#aaa");

    const sizeScale = d3.scaleLinear()
        .domain(d3.extent(graph.nodes.filter(d => d.type === 'option'), d => d.changed_globally))
        .range([8, 24]);
        
    const node = container.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(filteredNodes)
        .enter().append("circle")
        .attr("r", d => {
            if (d.type === 'option') {
                return sizeScale(d.changed_globally); // Scale size based on "changed_globally"
            }
            return 8; // Default size for other node types
        })
        .attr("fill", d => {
            if (d.type === 'concept') return "#1f77b4"; // Blue for concepts
            if (d.type === 'artifact') return "#ff7f0e"; // Orange for artifacts
            if (d.type === 'option') { // Green color based on "changed_internally"
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
        .filter(d => d.type === 'concept') // Only for concepts and artifacts
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
        
        const linkSizeScale = d3.scaleLinear()
            .domain(d3.extent(links, d => d.global_count))
            .range([1, 5]);
        
        const link = linkGroup.selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("stroke-width", d => { return linkSizeScale(d.global_count) })
            .attr("stroke", d => { return linkColorScale(d.internal_weight) })
            .attr("x1", d => filteredNodes.find(n => n.id === d.source).x)
            .attr("y1", d => filteredNodes.find(n => n.id === d.source).y)
            .attr("x2", d => filteredNodes.find(n => n.id === d.target).x)
            .attr("y2", d => filteredNodes.find(n => n.id === d.target).y)
            .style("visibility", visibilityState ? "visible" : "hidden"); // Initially hidden

        return link;
    }

    let conceptLinkGroup = container.append("g")
        .attr("class", "concept-links");
    let artifactLinkGroup = container.append("g")
        .attr("class", "artifact-links");
    let optionLinkGroup = container.append("g")
        .attr("class", "option-links");
    
    // Add links without force
    let conceptLink = addLinks(conceptLinkGroup, conceptLinks, state.showConceptLinks);
    let artifactLink = addLinks(artifactLinkGroup, artifactLinks, state.showArtifactLinks);
    let optionLink = addLinks(optionLinkGroup, optionLinks, state.showOptionLinks);

    attachTooltipListeners(conceptLink, artifactLink, optionLink)

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

        // Update artifact-to-artifact links
        artifactLink
            .attr("x1", d => filteredNodes.find(n => n.id === d.source).x)
            .attr("y1", d => filteredNodes.find(n => n.id === d.source).y)
            .attr("x2", d => filteredNodes.find(n => n.id === d.target).x)
            .attr("y2", d => filteredNodes.find(n => n.id === d.target).y);

        // Update option-to-option links
        optionLink
            .attr("x1", d => filteredNodes.find(n => n.id === d.source).x)
            .attr("y1", d => filteredNodes.find(n => n.id === d.source).y)
            .attr("x2", d => filteredNodes.find(n => n.id === d.target).x)
            .attr("y2", d => filteredNodes.find(n => n.id === d.target).y);

        // Dynamically update labels
        updateLabels(label, graph.nodes);
    });

    // Toggle concept link visibility
    document.getElementById('toggle-concept-links-checkbox').addEventListener('change', (event) => {
        state.showConceptLinks = event.target.checked;
        conceptLink.style("visibility", state.showConceptLinks ? "visible" : "hidden");
    });

    // Toggle artifact link visibility
    document.getElementById('toggle-artifact-links-checkbox').addEventListener('change', (event) => {
        state.showArtifactLinks = event.target.checked;
        artifactLink.style("visibility", state.showArtifactLinks ? "visible" : "hidden");
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

        // Clear existing concept, artifact, and option links
        conceptLinkGroup.selectAll("*").remove();
        artifactLinkGroup.selectAll("*").remove();
        optionLinkGroup.selectAll("*").remove();

        // Get new top-k concept, artifact, and option links
        conceptLinks = getLinks(graph, filteredNodes, 'concept-concept', state.topKValue, commitWindow);
        artifactLinks = getLinks(graph, filteredNodes, 'artifact-artifact', state.topKValue, commitWindow);
        optionLinks = getLinks(graph, filteredNodes, 'option-option', state.topKValue, commitWindow);

        // Append new concept, artifact, and option links
        conceptLink= addLinks(conceptLinkGroup, conceptLinks, state.showConceptLinks);
        artifactLink = addLinks(artifactLinkGroup, artifactLinks, state.showArtifactLinks);
        optionLink = addLinks(optionLinkGroup, optionLinks, state.showOptionLinks);

        attachTooltipListeners(conceptLink, artifactLink, optionLink)
     });

}

function loadGraphData(fileName, commitWindow) {
    const filePath = `/data/test_data/graph_data/${fileName}`;
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
