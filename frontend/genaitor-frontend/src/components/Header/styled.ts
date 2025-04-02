import styled from "styled-components"

export const ContentHeaderLayout = styled.header`
	display: flex;
	gap: 1rem;
	justify-content: space-between;
	height: 70px;
	border-bottom: 1px solid var(--gray);
	padding: 0.5rem 2rem;
	box-shadow: 0 4px 4px rgba(0, 0, 0, 0.25);

	.logo-content {
		background-color: var(--dark);
		padding: 0 10px;
		border-radius: 10px;
		display: flex;
		justify-content: center;
		align-items: center;

		img {
			object-fit: contain;
			width: 100%;
			height: 100%;
		}
	}

	@media (max-width: 768px) {
		padding: 0.5rem 1rem;
	}
`
