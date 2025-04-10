export function createLegend(svg) {

    const legendData = [
        { type: "Technology", color: "#1f77b4" },
        { type: "Configuration File", color: "#ff7f0e" },
        { type: "Configuration Option", color: "#2ca02c" }
    ];

    // Add legend for graph nodes
    const nodeLegendContainer = svg.append("g")
        .attr("class", "node-legend-container")
        .attr("transform", "translate(20, 240)");

    // Add a title label for the container
    nodeLegendContainer.append("text")
        .attr("x", 0)
        .attr("y", -30) // Adjust for spacing above the box
        .style("font-size", "16px")
        .style("font-weight", "bold")
        .style("fill", "#000")
        .text("Graph Nodes Legend");

    // Background border for the option legend
    nodeLegendContainer.append("rect")
        .attr("x", -10)
        .attr("y", -20)
        .attr("width", 220)
        .attr("height", 75) // Adjust height if needed
        .attr("stroke", "#000")
        .attr("fill", "none")
        .attr("stroke-width", 1.5)
        .attr("rx", 5) // Rounded corners
        .attr("ry", 5);

    // Add legend for discrete items
    const nodeLegend = nodeLegendContainer.append("g")
        .attr("class", "node-legend")
        .attr("transform", "translate(10, 0)");

    nodeLegend.selectAll("node-legend-item")
        .data(legendData)
        .enter().append("g")
        .attr("class", "node-legend-item")
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
    
        // Add a bordered group for the artifact node legend
    const conceptLegendContainer = svg.append("g")
        .attr("class", "concept-legend-container")
        .attr("transform", "translate(20, 360)");

    // Add a title label for the container
    conceptLegendContainer.append("text")
        .attr("x", 0)
        .attr("y", -30) // Adjust for spacing above the box
        .style("font-size", "16px")
        .style("font-weight", "bold")
        .style("fill", "#000")
        .text("Technology Encodings");

    // Background border for the artifact legend
    conceptLegendContainer.append("rect")
        .attr("x", -10)
        .attr("y", -20)
        .attr("width", 220)
        .attr("height", 70) // Adjust height if needed
        .attr("stroke", "#000")
        .attr("fill", "none")
        .attr("stroke-width", 1.5)
        .attr("rx", 5) // Rounded corners
        .attr("ry", 5);

    // Add the gradient legend
    const conceptGradientLegend = conceptLegendContainer.append("g")
        .attr("class", "concept-gradient-legend")
        .attr("transform", "translate(0, 30)");

    conceptGradientLegend.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 200)
        .attr("height", 10)
        .style("fill", "url(#concept-gradient)");

    conceptGradientLegend.append("text")
        .attr("x", 0)
        .attr("y", -5)
        .style("font-size", "12px")
        .style("fill", "#000")
        .text("Not Changed");

    conceptGradientLegend.append("text")
        .attr("x", 200)
        .attr("y", -5)
        .attr("text-anchor", "end")
        .style("font-size", "12px")
        .style("fill", "#000")
        .text("Frequently Changed");

    conceptGradientLegend.append("text")
        .attr("x", 0)
        .attr("y", -25)
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .style("fill", "#000")
        .text("Internal Changes");

    const conceptDefs = svg.append("defs");

    const conceptLinearGradient = conceptDefs.append("linearGradient")
        .attr("id", "concept-gradient")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%");

    conceptLinearGradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", "#d4f2ff");

    conceptLinearGradient.append("stop")
        .attr("offset", "50%")
        .attr("stop-color", "#7baede");

    conceptLinearGradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", "#001880");

    // Add a bordered group for the artifact node legend
    const artifactLegendContainer = svg.append("g")
        .attr("class", "artifact-legend-container")
        .attr("transform", "translate(20, 470)");

    // Add a title label for the container
    artifactLegendContainer.append("text")
        .attr("x", 0)
        .attr("y", -30) // Adjust for spacing above the box
        .style("font-size", "16px")
        .style("font-weight", "bold")
        .style("fill", "#000")
        .text("Artifact Encodings");

    // Background border for the artifact legend
    artifactLegendContainer.append("rect")
        .attr("x", -10)
        .attr("y", -20)
        .attr("width", 220)
        .attr("height", 70) // Adjust height if needed
        .attr("stroke", "#000")
        .attr("fill", "none")
        .attr("stroke-width", 1.5)
        .attr("rx", 5) // Rounded corners
        .attr("ry", 5);

    // Add the gradient legend
    const artifactGradientLegend = artifactLegendContainer.append("g")
        .attr("class", "artifact-gradient-legend")
        .attr("transform", "translate(0, 30)");

    artifactGradientLegend.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 200)
        .attr("height", 10)
        .style("fill", "url(#artifact-gradient)");

    artifactGradientLegend.append("text")
        .attr("x", 0)
        .attr("y", -5)
        .style("font-size", "12px")
        .style("fill", "#000")
        .text("Not Changed");

    artifactGradientLegend.append("text")
        .attr("x", 200)
        .attr("y", -5)
        .attr("text-anchor", "end")
        .style("font-size", "12px")
        .style("fill", "#000")
        .text("Frequently Changed");

    artifactGradientLegend.append("text")
        .attr("x", 0)
        .attr("y", -25)
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .style("fill", "#000")
        .text("Color: Changed Internal");

    const artifactDefs = svg.append("defs");

    const artifactLinearGradient = artifactDefs.append("linearGradient")
        .attr("id", "artifact-gradient")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%");

    artifactLinearGradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", "#ffe7ce");

    artifactLinearGradient.append("stop")
        .attr("offset", "50%")
        .attr("stop-color", "#ff9248");

    artifactLinearGradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", "#ff6700");
    

    // Add a bordered group for the option node legend
    const optionLegendContainer = svg.append("g")
        .attr("class", "option-legend-container")
        .attr("transform", "translate(20, 580)");

    // Add a title label for the container
    optionLegendContainer.append("text")
        .attr("x", 0)
        .attr("y", -30) // Adjust for spacing above the box
        .style("font-size", "16px")
        .style("font-weight", "bold")
        .style("fill", "#000")
        .text("Option Encodings");

    // Background border for the option legend
    optionLegendContainer.append("rect")
        .attr("x", -10)
        .attr("y", -20)
        .attr("width", 220)
        .attr("height", 70) // Adjust height if needed
        .attr("stroke", "#000")
        .attr("fill", "none")
        .attr("stroke-width", 1.5)
        .attr("rx", 5) // Rounded corners
        .attr("ry", 5);

    // Add the gradient legend
    const optionGradientLegend = optionLegendContainer.append("g")
        .attr("class", "option-gradient-legend")
        .attr("transform", "translate(0, 30)");

    optionGradientLegend.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 200)
        .attr("height", 10)
        .style("fill", "url(#option-gradient)");

    optionGradientLegend.append("text")
        .attr("x", 0)
        .attr("y", -5)
        .style("font-size", "12px")
        .style("fill", "#000")
        .text("Not Changed");

    optionGradientLegend.append("text")
        .attr("x", 200)
        .attr("y", -5)
        .attr("text-anchor", "end")
        .style("font-size", "12px")
        .style("fill", "#000")
        .text("Frequently Changed");

    optionGradientLegend.append("text")
        .attr("x", 0)
        .attr("y", -25)
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .style("fill", "#000")
        .text("Internal Changes");

    const optionDefs = svg.append("defs");

    const optionLinearGradient = optionDefs.append("linearGradient")
        .attr("id", "option-gradient")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%");

    optionLinearGradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", "#c7e9c0");

    optionLinearGradient.append("stop")
        .attr("offset", "50%")
        .attr("stop-color", "#41ab5d");

    optionLinearGradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", "#005a32");    
}
