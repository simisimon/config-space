const width = 800;
const height = 600;

const svg = d3.select("body")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

Promise.all([
    d3.json("commit_stats.json"),
    d3.json("changed_files.json")
]).then(([commitStats, changedFiles]) => {
    const data = {
        nodes: [],
        links: []
    };

    const developers = new Set();
    const artifacts = new Set();
    const links = new Set();

    commitStats.forEach(commit => {
        developers.add(commit.Contributor);
    });

    changedFiles.forEach(change => {
        artifacts.add(change["Changed File"]);
        links.add({ source: change.Contributor, target: change["Changed File"] });
    });

    developers.forEach(dev => {
        data.nodes.push({ id: dev, type: "developer" });
    });

    artifacts.forEach(file => {
        data.nodes.push({ id: file, type: "artifact" });
    });

    links.forEach(link => {
        data.links.push({ source: link.source, target: link.target });
    });

    const simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links).id(d => d.id))
        .force("charge", d3.forceManyBody().strength(-200))
        .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
        .selectAll("line")
        .data(data.links)
        .enter().append("line")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6);

    const node = svg.append("g")
        .selectAll("circle")
        .data(data.nodes)
        .enter().append("circle")
        .attr("r", 8)
        .attr("fill", d => d.type === "developer" ? "blue" : "red")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    const label = svg.append("g")
        .selectAll("text")
        .data(data.nodes)
        .enter().append("text")
        .attr("dx", 10)
        .attr("dy", ".35em")
        .text(d => d.id);

    simulation.on("tick", () => {
        link.attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node.attr("cx", d => d.x)
            .attr("cy", d => d.y);

        label.attr("x", d => d.x)
            .attr("y", d => d.y);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
});