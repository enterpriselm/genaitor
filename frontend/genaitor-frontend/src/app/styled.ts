import styled from "styled-components"

export const ContentSection = styled.section`
	display: flex;
	flex-direction: column;
	gap: 16px;
	width: 100%;
	padding: 1rem 2rem;
	height: 100%;
	min-height: calc(100vh - 70px);

	#tabs-content-option-genaitor {
		width: 100%;
		min-width: 720px;
		border-bottom-width: 2px;

		button {
			color: var(--gray);
			border: 0;
			background-color: transparent;
			border-bottom: 4px solid transparent;
			box-shadow: none;
			outline: none;

			&:hover,
			&:focus,
			&:active {
				color: var(--darker);
			}

			&.active {
				border-bottom-color: var(--primary-dark);
				color: var(--primary-dark);
			}
		}
	}

	@media (max-width: 768px) {
		padding: 1rem;
	}
`
