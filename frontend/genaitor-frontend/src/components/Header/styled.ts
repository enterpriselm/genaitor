import styled from "styled-components"

export const ContentHeaderLayout = styled.header`
	display: flex;
	gap: 1rem;
	justify-content: space-between;
	height: 70px;
	border-bottom: 1px solid var(--gray);
	padding: 0.5rem 2rem;
	box-shadow: 0 4px 4px rgba(0, 0, 0, 0.25);

	h1 {
		font-size: var(--tam-h3);
		font-weight: 700;
	}

	@media (max-width: 768px) {
		padding: 0.5rem 1rem;
	}
`
