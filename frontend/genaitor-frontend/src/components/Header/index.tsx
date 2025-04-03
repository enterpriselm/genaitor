"use client"
import Image from "next/image"
import { ContentHeaderLayout } from "./styled"
import Logo from "@/assets/images/logo.svg"
export default function Header() {
	return (
		<ContentHeaderLayout>
			<div className="logo-content">
				<Image src={Logo.src} height={100} width={100} alt="Genaitor Logo" />
			</div>
		</ContentHeaderLayout>
	)
}
