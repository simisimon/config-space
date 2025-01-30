export function createLegend(svg) {

    const legendContainer = svg.append("g")
        .attr("class", "legend-container")
        .attr("transform", "translate(20, 250)"); // Position the whole legend

    // Background rectangle for hover effect
    const legendBackground = legendContainer.append("rect")
        .attr("x", -20)
        .attr("y", -30)
        .attr("width", 240)
        .attr("height", 520) // Adjust height to fit all legend elements
        .attr("fill", "white") // Background color
        .attr("rx", 5) // Rounded corners
        .attr("ry", 5)
        .style("opacity", 1); // Keep background visible

    // --- Tooltip (Initially Hidden) ---
    const tooltipContainer = legendContainer.append("g")
        .attr("class", "legend-tooltip-container")
        .attr("transform", "translate(0, 180)") // Positioned below the legend
        .style("visibility", "hidden");

    // Use <foreignObject> to insert HTML content
    tooltipContainer.append("foreignObject")
        .attr("x", 0)
        .attr("y", 320)
        .attr("width", 230)
        .attr("height", 500) // Adjust height dynamically if needed
        .append("xhtml:div") // Use XHTML namespace to allow HTML inside SVG
        .html(`
        <p style="font-size: 12px; margin: 5;">
            <strong>Technology Nodes:</strong> represent technologies (e.g., programming languages, frameworks).
        </p>
        <p style="font-size: 12px; margin: 5;">
            <strong>Config File Node:</strong> represent configuration files (e.g., pom.xml or Dockerfile).
        </p>
        <p style="font-size: 12px; margin: 5;">
            <strong>Option Node:</strong> representing individual configuration options.
        </p>
        <p style="font-size: 12px; margin: 5;">
            <strong>(Option) Changed Interal:</strong> indicates the total number of times an options was changed within commit history of the project.
        </p>
        <p style="font-size: 12px; margin: 5;">
            <strong>(Option)Changed Global:</strong> indicates the total number of times an options was changed across the commit history of other projects.
        </p>
        <p style="font-size: 12px; margin: 5;">
            <strong>(Link) Changed Interal:</strong> indicates the total number of times the co-changed concepts/files/options changed within the commit history of the project.
        </p>
        <p style="font-size: 12px; margin: 5;">
            <strong>(Link) Changed Global:</strong> indicates the total number of times the co-changed concepts/files/options changed in the commit history of other projects.
        </p>
    `);

    // Show tooltip on hover
    legendContainer.on("mouseover", () => {
        tooltipContainer.style("visibility", "visible");
    }).on("mouseout", () => {
        tooltipContainer.style("visibility", "hidden");
    });

    const legendData = [
        { type: "Technology", color: "#1f77b4" },
        { type: "Configuration File", color: "#ff7f0e" },
        { type: "Configuration Option", color: "#2ca02c" }
    ];

    // Add legend for graph nodes
    const nodeLegendContainer = svg.append("g")
        .attr("class", "node-legend-container")
        .attr("transform", "translate(20, 270)");

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

    // Add a bordered group for the option node legend
    const optionLegendContainer = svg.append("g")
        .attr("class", "option-legend-container")
        .attr("transform", "translate(20, 390)");

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
        .attr("height", 150) // Adjust height if needed
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
        .text("Color: Changed Internal");

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

    // Add size legend for changed_globally
    const sizeLegend = optionLegendContainer.append("g")
        .attr("class", "size-legend")
        .attr("transform", "translate(0, 100)");

    const sizeScale = d3.scaleLinear()
        .range([8, 20]); // Corresponding circle sizes

    // Generate representative values for the size legend
    const sizeLegendValues = sizeScale.ticks(3); // Adjust number of ticks as needed

    // Add a title for the size legend
    sizeLegend.append("text")
        .attr("x", 0)
        .attr("y", -30)
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .style("fill", "#000")
        .text("Node Size: Changed Global");

    // Add size legend circles and labels
    sizeLegend.selectAll(".size-legend-item")
        .data(sizeLegendValues)
        .enter().append("g")
        .attr("class", "size-legend-item")
        .attr("transform", (d, i) => `translate(${i * 50}, 0)`) // Space circles horizontally
        .each(function (d) {
            const item = d3.select(this);

            // Add a circle representing the size
            item.append("circle")
                .attr("cx", 10)
                .attr("cy", 0)
                .attr("r", sizeScale(d)) // Scale size based on value
                .attr("fill", "#2ca02c"); // Match the color of option nodes
        });
    
    
    // Add a bordered group for the option node legend
    const linkLegendContainer = svg.append("g")
        .attr("class", "link-legend-container")
        .attr("transform", "translate(20, 590)");

    // Add a title label for the container
    linkLegendContainer.append("text")
        .attr("x", 0)
        .attr("y", -30) // Adjust for spacing above the box
        .style("font-size", "16px")
        .style("font-weight", "bold")
        .style("fill", "#000")
        .text("Link Encodings");

    // Background border for the option legend
    linkLegendContainer.append("rect")
        .attr("x", -10)
        .attr("y", -20)
        .attr("width", 220)
        .attr("height", 160) // Adjust height if needed
        .attr("stroke", "#000")
        .attr("fill", "none")
        .attr("stroke-width", 1.5)
        .attr("rx", 5) // Rounded corners
        .attr("ry", 5);
    
    
    // Add the gradient legend
    const linkGradientLegend = linkLegendContainer.append("g")
        .attr("class", "link-gradient-legend")
        .attr("transform", "translate(0, 30)");

    linkGradientLegend.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 200)
        .attr("height", 10)
        .style("fill", "url(#link-gradient)");

    linkGradientLegend.append("text")
        .attr("x", 0)
        .attr("y", -5)
        .style("font-size", "12px")
        .style("fill", "#000")
        .text("Not Changed");

    linkGradientLegend.append("text")
        .attr("x", 200)
        .attr("y", -5)
        .attr("text-anchor", "end")
        .style("font-size", "12px")
        .style("fill", "#000")
        .text("Frequently Changed");

    linkGradientLegend.append("text")
        .attr("x", 0)
        .attr("y", -25)
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .style("fill", "#000")
        .text("Color: Changed Internal");

    const linkDefs = svg.append("defs");

    const linkLinearGradient = linkDefs.append("linearGradient")
        .attr("id", "link-gradient")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%");

    linkLinearGradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", "#ffbaba");

    linkLinearGradient.append("stop")
        .attr("offset", "50%")
        .attr("stop-color", "#ff5252");

    linkLinearGradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", "#a70000");
    
    // Define the scale for link thickness
    const linkSizeScale = d3.scaleLinear()
        .domain([1, 10]) // Adjust domain based on expected link weights
        .range([1, 5]); // Line thickness range

    // Example values for the legend (representing increasing link thickness)
    const linkSizes = [1, 5, 10];

    // Add link lines for legend
    const linkLegendSize = linkLegendContainer.append("g")
        .attr("class", "link-size-legend")
        .attr("transform", "translate(0, 85)");

    linkLegendSize.append("text")
        .attr("x", 0)
        .attr("y", -20)
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .style("fill", "#000")
        .text("Link Size: Changed Global");
    
    linkLegendSize.selectAll(".link-legend-item")
        .data(linkSizes)
        .enter().append("g")
        .attr("class", "link-legend-item")
        .attr("transform", (d, i) => `translate(0, ${i * 20})`)
        .each(function (d) {
            const item = d3.select(this);
            // Add a line representing link size
            item.append("line")
                .attr("x1", 0)
                .attr("y1", 0)
                .attr("x2", 50)
                .attr("y2", 0)
                .attr("stroke", "#a70000")
                .attr("stroke-width", linkSizeScale(d));

        });
    
    
    // Add a single label for the whole legend
    linkLegendContainer.append("text")
        .attr("x", 60)
        .attr("y", 90)
        .style("font-size", "12px")
        .style("fill", "#000")
        .text("Less Changed");
    
    // Add a single label for the whole legend
    linkLegendContainer.append("text")
        .attr("x", 60)
        .attr("y", 130)
        .style("font-size", "12px")
        .style("fill", "#000")
        .text("Frequently Changed");
}
