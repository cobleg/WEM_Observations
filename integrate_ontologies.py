import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import argparse

# Import our semantic translator modules
from electricity_market_semantic_translator import ElectricityMarketSemanticTranslator
from market_ontology_integrator import MarketOntologyIntegrator, process_wem_data_with_ontologies

def main():
    """
    Main function to demonstrate automatic translation of electricity market data
    into semantic statements.
    """
    parser = argparse.ArgumentParser(description='Translate WEM market data into semantic statements')
    parser.add_argument('--db_path', type=str, default='F:\\wem_data_processor_0_1\\src\\data\\db\\wem_data.db',
                        help='Path to the WEM database')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output files')
    parser.add_argument('--start_date', type=str, default='2024-01-21',
                        help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-01-22',
                        help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--method', type=str, choices=['basic', 'ontology'], default='ontology',
                        help='Method to use: basic or ontology-based')
    parser.add_argument('--services', type=str, nargs='+', 
                        default=['energy', 'contingencyRaise', 'contingencyLower', 'regulationRaise', 'regulationLower', 'rocof'],
                        help='Market services to analyze')
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='Z-score threshold for event detection')
    parser.add_argument('--min_change', type=float, default=1.0,
                        help='Minimum absolute change to consider significant')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print some info
    print(f"Processing WEM data from {args.start_date} to {args.end_date}")
    print(f"Using method: {args.method}")
    print(f"Output directory: {args.output_dir}")
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        print(f"Error: Database file not found at {args.db_path}")
        return
    
    # Check for available data
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()
    
    try:
        # Check for available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Available tables: {', '.join([t[0] for t in tables])}")
        
        # Check for date range
        cursor.execute("SELECT MIN(interval_time), MAX(interval_time) FROM dispatch_prices")
        min_date, max_date = cursor.fetchone()
        print(f"Available data range: {min_date} to {max_date}")
        
        # Check for services
        cursor.execute("SELECT DISTINCT market_service FROM dispatch_prices")
        available_services = [row[0] for row in cursor.fetchall()]
        print(f"Available market services: {', '.join(available_services)}")
        
        # Check for facilities
        cursor.execute("SELECT COUNT(DISTINCT facility_name) FROM dispatch_quantities")
        facility_count = cursor.fetchone()[0]
        print(f"Number of facilities: {facility_count}")
    except Exception as e:
        print(f"Error checking database: {e}")
        conn.close()
        return
    
    conn.close()
    
    # Process the data based on selected method
    if args.method == 'basic':
        process_basic_method(args)
    else:
        process_ontology_method(args)


def process_basic_method(args):
    """Process data using the basic semantic translator method."""
    print("\nProcessing with basic semantic translator...")
    
    # Initialize translator
    translator = ElectricityMarketSemanticTranslator(args.db_path)
    
    # Process each requested service
    for service in args.services:
        print(f"Processing {service} service...")
        translator.process_market_service(
            service_type=service,
            date_range=(args.start_date, args.end_date),
            threshold=args.threshold,
            min_value_change=args.min_change
        )
    
    # Detect correlations between events
    print("Detecting event correlations...")
    translator.detect_correlations()
    
    # Generate and save outputs
    statements = translator.generate_human_readable_statements()
    print(f"Generated {len(statements)} semantic statements")
    
    if len(statements) > 0:
        print("\nSample statements:")
        for i, statement in enumerate(statements[:5], 1):
            print(f"{i}. {statement}")
        print("...")
    
    # Save outputs
    output_ttl = os.path.join(args.output_dir, "wem_semantic_graph_basic.ttl")
    output_txt = os.path.join(args.output_dir, "wem_semantic_statements_basic.txt")
    
    translator.save_graph(output_ttl)
    translator.save_statements(output_txt)
    
    print(f"\nRDF graph saved to: {output_ttl}")
    print(f"Human-readable statements saved to: {output_txt}")
    
    # Close connection
    translator.close()


def process_ontology_method(args):
    """Process data using the ontology-based method."""
    print("\nProcessing with ontology-based translator (FIBO & Event Model F)...")
    
    # Use the integrated ontology processor
    statements = process_wem_data_with_ontologies(
        args.db_path,
        output_dir=args.output_dir
    )
    
    if len(statements) > 0:
        print("\nSample statements:")
        for i, statement in enumerate(statements[:5], 1):
            print(f"{i}. {statement}")
        print("...")
    
    output_ttl = os.path.join(args.output_dir, "wem_semantic_graph.ttl")
    output_txt = os.path.join(args.output_dir, "wem_semantic_statements.txt")
    output_jsonld = os.path.join(args.output_dir, "wem_semantic_graph.jsonld")
    
    print(f"\nRDF graph saved to: {output_ttl}")
    print(f"JSON-LD export saved to: {output_jsonld}")
    print(f"Human-readable statements saved to: {output_txt}")


def analyze_facility_interactions(db_path, facility_name, date_range=None):
    """
    Special analysis function to analyze a specific facility's interactions
    with market prices and other facilities.
    """
    print(f"\nPerforming deep analysis of facility: {facility_name}")
    
    conn = sqlite3.connect(db_path)
    
    # Build date range filter if provided
    date_filter = ""
    if date_range:
        start_date, end_date = date_range
        date_filter = f" AND interval_time BETWEEN '{start_date}' AND '{end_date}'"
    
    # Get facility data
    facility_query = f"""
    SELECT interval_time, market_service, quantity 
    FROM dispatch_quantities
    WHERE facility_name = '{facility_name}'{date_filter}
    ORDER BY interval_time, market_service
    """
    
    facility_df = pd.read_sql_query(facility_query, conn)
    
    if facility_df.empty:
        print(f"No data found for facility {facility_name}")
        conn.close()
        return []
    
    print(f"Found {len(facility_df)} records for {facility_name}")
    
    # Get corresponding price data
    unique_services = facility_df['market_service'].unique()
    unique_timestamps = facility_df['interval_time'].unique()
    
    price_query = f"""
    SELECT interval_time, market_service, price 
    FROM dispatch_prices
    WHERE market_service IN ({', '.join(['?']*len(unique_services))})
    AND interval_time IN ({', '.join(['?']*len(unique_timestamps))})
    ORDER BY interval_time, market_service
    """
    
    price_df = pd.read_sql_query(price_query, conn, params=list(unique_services) + list(unique_timestamps))
    
    # Merge the data
    merged_df = pd.merge(
        facility_df, 
        price_df, 
        on=['interval_time', 'market_service'],
        how='inner'
    )
    
    # Calculate correlations between quantity and price
    correlations = {}
    insights = []
    
    for service in unique_services:
        service_data = merged_df[merged_df['market_service'] == service]
        if len(service_data) > 5:  # Need enough data points
            # Calculate correlation
            corr = service_data['quantity'].corr(service_data['price'])
            correlations[service] = corr
            
            # Generate insight
            if abs(corr) > 0.5:
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) > 0.7 else "moderate"
                insight = (
                    f"There is a {strength} {direction} correlation ({corr:.2f}) between "
                    f"{facility_name}'s dispatch quantity and the {service} price."
                )
                insights.append(insight)
    
    # Get other facilities that appear to be correlated with this one
    related_facilities_query = f"""
    WITH facility_data AS (
        SELECT interval_time, market_service, quantity
        FROM dispatch_quantities
        WHERE facility_name = '{facility_name}'{date_filter}
    )
    SELECT dq.interval_time, dq.market_service, dq.facility_name, dq.quantity, fd.quantity as target_quantity
    FROM dispatch_quantities dq
    JOIN facility_data fd ON dq.interval_time = fd.interval_time AND dq.market_service = fd.market_service
    WHERE dq.facility_name != '{facility_name}'
    ORDER BY dq.interval_time, dq.market_service, dq.facility_name
    """
    
    related_df = pd.read_sql_query(related_facilities_query, conn)
    
    if not related_df.empty:
        # Group by facility and service
        facility_correlations = {}
        
        for facility in related_df['facility_name'].unique():
            facility_correlations[facility] = {}
            
            for service in related_df['market_service'].unique():
                service_data = related_df[
                    (related_df['facility_name'] == facility) & 
                    (related_df['market_service'] == service)
                ]
                
                if len(service_data) > 5:  # Need enough data points
                    # Calculate correlation
                    corr = service_data['quantity'].corr(service_data['target_quantity'])
                    facility_correlations[facility][service] = corr
                    
                    # Generate insight for strong correlations
                    if abs(corr) > 0.7:
                        direction = "positive" if corr > 0 else "negative"
                        insight = (
                            f"{facility_name} and {facility} show a strong {direction} correlation ({corr:.2f}) "
                            f"in their {service} dispatch quantities, suggesting they may be responding to similar market signals."
                        )
                        insights.append(insight)
    
    conn.close()
    return insights


if __name__ == "__main__":
    main()
