import { useState } from "react";
import { useForm } from "react-hook-form";
import axios from "axios";

export default function PreReleaseForm() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm();
  const [message, setMessage] = useState("");

  const onSubmit = async (data) => {
    try {
      await axios.post("http://localhost:5000/api/pre-release", data);
      setMessage("Obrigado por se inscrever! Você receberá um e-mail em breve.");
    } catch (error) {
      setMessage("Erro ao enviar. Tente novamente mais tarde.");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 p-4">
      <div className="bg-white p-8 rounded-xl shadow-lg max-w-md w-full">
        <h2 className="text-2xl font-bold text-center mb-4">GenAitor - Pré-Release</h2>
        <p className="text-gray-600 text-center mb-6">
          Inscreva-se para obter acesso ao repositório no início de março.
        </p>
        <p className="text-gray-700 text-center mb-6">
          O GenAitor é uma plataforma avançada para a criação e automação de agentes de IA.
          Com suporte para múltiplos provedores de LLM (Gemini, Claude, OpenAI, DeepSeek, ou personalizados),
          ele permite configurar e ajustar modelos de linguagem, criar e atribuir tarefas específicas a agentes,
          definir fluxos de execução por meio de um orquestrador e fornecer dados de entrada para processamento.
          O sistema gera respostas em formatos específicos e gerencia erros automaticamente, garantindo eficiência
          e flexibilidade para empresas e desenvolvedores que buscam otimizar fluxos de trabalho com IA generativa.
        </p>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <input
            {...register("name", { required: "Nome é obrigatório" })}
            className="w-full p-3 border rounded"
            placeholder="Nome"
          />
          {errors.name && <p className="text-red-500 text-sm">{errors.name.message}</p>}

          <input
            {...register("email", {
              required: "E-mail é obrigatório",
               pattern: {
                value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/i,
                message: "E-mail inválido",
              },
            })}
            className="w-full p-3 border rounded"
            placeholder="E-mail"
          />
          {errors.email && <p className="text-red-500 text-sm">{errors.email.message}</p>}

          <input
            {...register("company")}
            className="w-full p-3 border rounded"
            placeholder="Empresa (opcional)"
          />

          <input
            {...register("role")}
            className="w-full p-3 border rounded"
            placeholder="Cargo (opcional)"
          />

          <input
            {...register("country", { required: "País é obrigatório" })}
            className="w-full p-3 border rounded"
            placeholder="País"
          />
          {errors.country && <p className="text-red-500 text-sm">{errors.country.message}</p>}

          <button type="submit" className="w-full bg-blue-600 text-white p-3 rounded font-bold">
            Inscrever-se
          </button>
        </form>
        {message && <p className="text-center mt-4 text-gray-700">{message}</p>}
      </div>
    </div>
  );
}
