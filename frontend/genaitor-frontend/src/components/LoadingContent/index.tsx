"use client"
import { Spinner } from "react-bootstrap"
import { LoadingContentStyled } from "./styled"
export default function LoadingContent() {
	return (
		<LoadingContentStyled>
			<h6>Loading Analysis...</h6>
			<Spinner />
		</LoadingContentStyled>
	)
}
