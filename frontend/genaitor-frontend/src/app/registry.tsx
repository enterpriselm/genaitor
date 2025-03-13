"use client"

import React from "react"
import { useServerInsertedHTML } from "next/navigation"
import { ServerStyleSheet, StyleSheetManager } from "styled-components"

export default function StyledComponentsRegistry({
	children,
}: Readonly<{
	children: Readonly<React.ReactNode>
}>) {
	// Only create stylesheet once with lazy initial state
	// x-ref: https://reactjs.org/docs/hooks-reference.html#lazy-initial-state
	const styledComponentsStyleSheet = new ServerStyleSheet()

	useServerInsertedHTML(() => {
		const styles = styledComponentsStyleSheet.getStyleElement()
		return <>{styles}</>
	})

	if (typeof window !== "undefined") return <>{children}</>

	return (
		<StyleSheetManager sheet={styledComponentsStyleSheet.instance}>
			{children}
		</StyleSheetManager>
	)
}
