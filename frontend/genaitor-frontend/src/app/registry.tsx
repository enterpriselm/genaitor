"use client"

import React, { useState } from "react"
import { useServerInsertedHTML } from "next/navigation"
import { ServerStyleSheet, StyleSheetManager } from "styled-components"

export default function StyledComponentsRegistry({
	children,
}: Readonly<{
	children: Readonly<React.ReactNode>
}>) {
	// Usar useState para garantir que o stylesheet seja criado apenas uma vez
	const [styledComponentsStyleSheet] = useState(() => new ServerStyleSheet())

	useServerInsertedHTML(() => {
		const styles = styledComponentsStyleSheet.getStyleElement()
		styledComponentsStyleSheet.instance.clearTag()
		return <>{styles}</>
	})

	// Verificar se estamos no cliente de uma maneira que não cause problemas de hidratação
	const isClient = typeof window !== "undefined"

	if (isClient) {
		// No cliente, apenas renderizar os filhos
		return <>{children}</>
	}

	// No servidor, usar o StyleSheetManager
	return (
		<StyleSheetManager sheet={styledComponentsStyleSheet.instance}>
			{children}
		</StyleSheetManager>
	)
}
