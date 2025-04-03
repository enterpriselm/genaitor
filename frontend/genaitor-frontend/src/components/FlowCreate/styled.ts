import styled from "styled-components"

export const FlowCreateContainer = styled.div`
	display: flex;
	flex-direction: column;
	gap: 16px;
	width: 100%;
	padding: 1rem;

	.dropdown-agents {
		width: 100%;

		button {
			border: 1px solid var(--gray);
			background-color: var(--white);
			color: var(--dark);
			border-radius: 5px;
			cursor: pointer;
			display: flex;
			justify-content: space-between;
			width: 100%;
			transition: all 0.5s;

			p {
				transition: all 0.5s;
			}

			&::after,
			&::before {
				content: "";
				display: none;
			}

			&:hover,
			&:focus,
			&:active {
				color: var(--white) !important;
				background-color: var(--primary);
				outline: none;
				box-shadow: none;

				p {
					color: var(--white) !important;
				}
			}
		}

		.dropdown-menu {
			overflow: auto;
			max-height: 200px;

			.content-input {
				padding: 0 0.625rem;

				input {
					border: none;
					border-bottom: 2px solid var(--gray);
					border-radius: 0;

					&:focus,
					&:active {
						outline: none;
						box-shadow: none;
						border-bottom: 2px solid var(--primary);
					}
				}
			}
		}
	}

	.response-content {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}
`
