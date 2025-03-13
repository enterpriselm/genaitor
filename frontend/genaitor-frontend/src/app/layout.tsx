"use client"

import { Nunito } from "next/font/google"
import "bootstrap/dist/css/bootstrap.min.css"
import "@/assets/css/globals.css"
import "react-toastify/dist/ReactToastify.css"
import StyledComponentsRegistry from "./registry"
import { ToastContainer } from "react-toastify"
import React from "react"
import Header from "@/components/Header"

const nunito = Nunito({
	subsets: ["latin"],
	variable: "--font-nunito",
})

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode
}>) {
	React.useEffect(() => {
		// eslint-disable-next-line @typescript-eslint/no-require-imports
		require("bootstrap/dist/js/bootstrap.bundle.min.js")
	}, [])

	return (
		<html lang="en">
			<head>
				<title>Genaitor Frontend</title>
				<link rel="icon" href="/favicon.ico" />
			</head>
			<body className={`${nunito.variable}`}>
				<ToastContainer
					position="top-right"
					hideProgressBar={false}
					newestOnTop={false}
					closeOnClick
					pauseOnFocusLoss
					draggable
					pauseOnHover
					theme="colored"
					limit={1}
				/>
				<StyledComponentsRegistry>
					<main>
						<Header />
						{children}
					</main>
				</StyledComponentsRegistry>
			</body>
		</html>
	)
}
