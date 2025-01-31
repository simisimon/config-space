const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const dataDir = path.join(__dirname, 'data/graph_data');

app.use(express.static(__dirname));

app.get('/files', (req, res) => {
    fs.readdir(dataDir, (err, files) => {
        if (err) {
            return res.status(500).send('Unable to scan directory');
        }
        res.json(files);
    });
});

app.get('/data/test_data/graph_data/:filename', (req, res) => {
    const filePath = path.join(dataDir, req.params.filename);
    if (fs.existsSync(filePath)) {
        res.sendFile(filePath);
    } else {
        res.status(404).send('File not found');
    }
});


const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});