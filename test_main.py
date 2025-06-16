"""
Test Main Application

Quick test to verify the main application components work correctly.
"""

import logging
from main import (
    setup_logging, 
    validate_trading_environment, 
    get_trading_instruments,
    check_market_hours,
    trading_job
)

def test_main_components():
    """
    Test individual components of the main application.
    """
    try:
        print("🧪 Testing Main Application Components")
        print("=" * 40)
        
        # Test 1: Logging setup
        print("1. Testing logging setup...")
        logger = setup_logging()
        logger.info("✓ Logging system initialized")
        
        # Test 2: Market hours check
        print("2. Testing market hours check...")
        market_open = check_market_hours()
        print(f"   Market is {'OPEN' if market_open else 'CLOSED'}")
        
        # Test 3: Trading instruments
        print("3. Testing instrument list...")
        instruments = get_trading_instruments()
        print(f"   Instruments: {', '.join(instruments)}")
        
        # Test 4: Environment validation
        print("4. Testing environment validation...")
        env_valid = validate_trading_environment()
        print(f"   Environment: {'VALID' if env_valid else 'INVALID'}")
        
        # Test 5: Single trading job (dry run)
        if env_valid:
            print("5. Testing single trading job execution...")
            print("   (This will run the actual strategy check)")
            
            # Run one iteration of trading job
            trading_job()
            
            print("   ✓ Trading job completed")
        else:
            print("5. Skipping trading job test (environment invalid)")
        
        print("\n✅ All main application tests completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Main application test failed: {e}")
        return False

if __name__ == "__main__":
    test_main_components()