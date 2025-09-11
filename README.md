A tool to aid in the logistics of organising airport transfers.

This tool was created to solve a real problem at a school I work at. The school organises study trips for groups of students, and provides them accommodation with local families. The groups are then met at the airport and put into pre-booked taxis to take them to their accommodation. When booking the taxis, the destinations need to be provided, with a maximum of six passengers per taxi. Arranging these routes manually can take several hours. 

I decided to build a tool which leverages Google's Routes API to calculate real driving distances, then uses Google's OR-Tools to calculate the optimal routing given the constraints.

**Roadmap:**
- Current Stage: implementing the Routes API and testing with real data
- Next stage: creating a user-friendly interface for use by non-technical stakeholders
- Future steps: host the tool on the web so it can be used from anywhere
