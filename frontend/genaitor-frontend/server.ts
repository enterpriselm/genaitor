import axios from "axios"

export const service = axios.create({
	baseURL: process.env.NEXT_PUBLIC_BACKEND,
	headers: {
		"Content-Type": "application/json",
	},
})
