const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express();
app.use(cors());
app.use(bodyParser.json());

app.post("/api/pre-release", (req, res) => {
  const { name, email, company, role, country } = req.body;

  if (!name || !email || !country) {
    return res.status(400).json({ message: "Nome, E-mail e País são obrigatórios." });
  }

  console.log("Novo registro:", { name, email, company, role, country });

  res.status(200).json({ message: "Obrigado por se inscrever! Você receberá um e-mail em breve." });
});

app.listen(5000, () => {
  console.log("Servidor rodando na porta 5000");
});
