import { Nunito } from "next/font/google"
import "bootstrap/dist/css/bootstrap.min.css"
import "@/assets/css/globals.css"
import "react-toastify/dist/ReactToastify.css"
import StyledComponentsRegistry from "./registry"
import { ToastContainer } from "react-toastify"
import React from "react"
import Header from "@/components/Header"
import BootstrapClient from "@/components/BootstrapClient"

const nunito = Nunito({
	subsets: ["latin"],
	variable: "--font-nunito",
})

export const metadata = {
	title: "Genaitor Frontend",
	description: "Frontend application for Genaitor",
}

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode
}>) {
	return (
		<html lang="en">
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
					<BootstrapClient />
					<main>
						<Header />
						{children}
					</main>
				</StyledComponentsRegistry>
			</body>
		</html>
	)
}
