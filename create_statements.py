import sqlite3
import pandas as pd
from electricity_market_semantic_translator import ElectricityMarketSemanticTranslator

# Set up database connection
db_path = "F:\\wem_data_processor_0_1\\src\\data\\db\\wem_data.db"

# Initialize the semantic translator
translator = ElectricityMarketSemanticTranslator(db_path)

# Example 1: Process a specific day and service
print("Example 1: Processing energy data for January 21, 2024")
translator.process_market_service(
    service_type='energy',
    date_range=("2024-01-21", "2024-01-22"),
    threshold=2.0,  # Z-score threshold for detecting significant events
    min_value_change=5.0  # Minimum absolute change to consider significant
)

# Generate and print statements
statements = translator.generate_human_readable_statements()
print(f"Generated {len(statements)} statements:")
for i, statement in enumerate(statements[:5], 1):
    print(f"{i}. {statement}")
print("...")

# Example 2: Process all services for a time period
print("\nExample 2: Processing all services for January 21-22, 2024")
translator = ElectricityMarketSemanticTranslator(db_path)  # Fresh instance
translator.process_all_services(
    date_range=("2024-01-21", "2024-01-22"),
    threshold=2.5,
    min_value_change=1.0
)

# Generate and print statements
all_statements = translator.generate_human_readable_statements()
print(f"Generated {len(all_statements)} statements across all services:")
for i, statement in enumerate(all_statements[:5], 1):
    print(f"{i}. {statement}")
print("...")

# Example 3: Analyze a specific facility
print("\nExample 3: Analyzing ALBANY_WF1 facility")
# Check if we have data for this facility
conn = sqlite3.connect(db_path)
facility_query = """
SELECT COUNT(*) as count
FROM dispatch_quantities
WHERE facility_name = 'ALBANY_WF1'
"""
result = pd.read_sql_query(facility_query, conn)
if result['count'][0] > 0:
    translator = ElectricityMarketSemanticTranslator(db_path)  # Fresh instance
    
    # Get facility data
    facility_data_query = """
    SELECT interval_time, quantity, market_service
    FROM dispatch_quantities
    WHERE facility_name = 'ALBANY_WF1'
    ORDER BY interval_time, market_service
    LIMIT 5
    """
    facility_data = pd.read_sql_query(facility_data_query, conn)
    print("Sample ALBANY_WF1 data:")
    print(facility_data)
    
    # Process facility data
    translator.process_market_service(
        service_type='energy',  # Assuming energy service
        date_range=None,  # All available dates
        threshold=1.5,  # Lower threshold to catch more events for this facility
        min_value_change=0.5  # Lower minimum change for wind farm (more variable)
    )
    
    # Get facility-specific statements
    facility_statements = []
    all_statements = translator.generate_human_readable_statements()
    for statement in all_statements:
        if "ALBANY_WF1" in statement:
            facility_statements.append(statement)
    
    print(f"\nGenerated {len(facility_statements)} statements for ALBANY_WF1:")
    for i, statement in enumerate(facility_statements[:5], 1):
        print(f"{i}. {statement}")
    print("...")
else:
    print("No data found for ALBANY_WF1 facility in the database.")

conn.close()

# Example 4: Export to different formats
print("\nExample 4: Exporting semantic data to different formats")
translator = ElectricityMarketSemanticTranslator(db_path)
translator.process_market_service('energy', date_range=("2024-01-21", "2024-01-21"))

# Save RDF graph in Turtle format
turtle_path = "wem_semantic_graph.ttl"
translator.save_graph(turtle_path)
print(f"Saved RDF graph to {turtle_path}")

# Save human-readable statements
statements_path = "wem_semantic_statements.txt"
translator.save_statements(statements_path)
print(f"Saved human-readable statements to {statements_path}")

# Export as JSON-LD (needs additional code)
# This would be an extension point for integration with other systems

# Clean up
translator.close()
