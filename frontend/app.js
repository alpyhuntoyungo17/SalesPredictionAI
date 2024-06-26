const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;

// Middleware untuk mengurai permintaan JSON
app.use(bodyParser.json());

// Route untuk prediksi
app.post('/predict', async (req, res) => {
    try {
        const data = req.body;

        // Mengirim permintaan ke server Flask
        const response = await axios.post('http://localhost:5000/predict', data);

        // Mengembalikan hasil prediksi ke klien
        res.json(response.data);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Error in making prediction' });
    }
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});

