"""
Parameterized Tests for Performance ML System
Phase 1: Data Collection & Preprocessing - Step 1
"""

import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import time

class NetworkCondition(Enum):
    WIFI = "wifi"
    FOUR_G = "4g"
    THREE_G = "3g"
    SLOW_3G = "slow_3g"

class BrowserType(Enum):
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"

class TestType(Enum):
    API = "api"
    FRONTEND = "frontend"
    LOAD = "load"
    STRESS = "stress"

@dataclass
class TestConfiguration:
    """Configuration for a single performance test"""
    test_id: str
    test_type: TestType
    concurrent_users: int
    spawn_rate: float  # users per second
    duration: int  # seconds
    network_condition: NetworkCondition
    browser: Optional[BrowserType] = None
    api_endpoints: Optional[List[str]] = None
    custom_headers: Optional[Dict[str, str]] = None
    payload_size: Optional[int] = None  # bytes
    target_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        config = asdict(self)
        config['test_type'] = self.test_type.value
        config['network_condition'] = self.network_condition.value
        if self.browser:
            config['browser'] = self.browser.value
        return config

class TestParameterizer:
    """Generates parameterized test configurations"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.api_endpoints = [
            "/api/health",
            "/api/data/test",
            "/api/search?q=performance",
            "/api/users?page=1&limit=10",
            "/api/upload",
            "/api/slow",
            "/api/memory"
        ]
    
    def generate_api_test_configs(self) -> List[TestConfiguration]:
        """Generate API-focused test configurations"""
        configs = []
        
        # Different user load scenarios
        user_scenarios = [
            (10, 1.0),   # Light load
            (50, 2.0),   # Medium load
            (100, 5.0),  # High load
            (200, 10.0), # Very high load
            (500, 20.0), # Stress test
        ]
        
        # Network conditions
        networks = [NetworkCondition.WIFI, NetworkCondition.FOUR_G, NetworkCondition.THREE_G]
        
        for users, spawn_rate in user_scenarios:
            for network in networks:
                for endpoint in self.api_endpoints:
                    config = TestConfiguration(
                        test_id=f"api_{users}u_{network.value}_{endpoint.replace('/', '_')}",
                        test_type=TestType.API,
                        concurrent_users=users,
                        spawn_rate=spawn_rate,
                        duration=300,  # 5 minutes
                        network_condition=network,
                        api_endpoints=[endpoint],
                        target_url=f"{self.base_url}{endpoint}"
                    )
                    configs.append(config)
        
        return configs
    
    def generate_frontend_test_configs(self) -> List[TestConfiguration]:
        """Generate frontend-focused test configurations"""
        configs = []
        
        # Browser scenarios
        browsers = [BrowserType.CHROME, BrowserType.FIREFOX, BrowserType.SAFARI]
        
        # Network conditions for frontend
        networks = [NetworkCondition.WIFI, NetworkCondition.FOUR_G, NetworkCondition.SLOW_3G]
        
        # Different page scenarios
        pages = [
            "/",  # Homepage
            "/?heavy=true",  # Heavy content
        ]
        
        for browser in browsers:
            for network in networks:
                for page in pages:
                    config = TestConfiguration(
                        test_id=f"frontend_{browser.value}_{network.value}_{page.replace('/', '_')}",
                        test_type=TestType.FRONTEND,
                        concurrent_users=1,  # Frontend tests are typically single-user
                        spawn_rate=0.1,
                        duration=60,  # 1 minute per test
                        network_condition=network,
                        browser=browser,
                        target_url=f"{self.base_url}{page}"
                    )
                    configs.append(config)
        
        return configs
    
    def generate_load_test_configs(self) -> List[TestConfiguration]:
        """Generate comprehensive load test configurations"""
        configs = []
        
        # Progressive load testing
        load_scenarios = [
            (10, 1.0, 120),    # Ramp up
            (50, 2.0, 300),    # Sustained load
            (100, 5.0, 600),   # Peak load
            (200, 10.0, 900),  # High load
            (500, 20.0, 1200), # Stress test
        ]
        
        for users, spawn_rate, duration in load_scenarios:
            config = TestConfiguration(
                test_id=f"load_{users}u_{spawn_rate}s",
                test_type=TestType.LOAD,
                concurrent_users=users,
                spawn_rate=spawn_rate,
                duration=duration,
                network_condition=NetworkCondition.WIFI,
                api_endpoints=self.api_endpoints,
                target_url=self.base_url
            )
            configs.append(config)
        
        return configs
    
    def generate_stress_test_configs(self) -> List[TestConfiguration]:
        """Generate stress test configurations to find breaking points"""
        configs = []
        
        # Stress scenarios
        stress_scenarios = [
            (1000, 50.0, 1800),   # High stress
            (2000, 100.0, 2400),  # Very high stress
            (5000, 200.0, 3600),  # Extreme stress
        ]
        
        for users, spawn_rate, duration in stress_scenarios:
            config = TestConfiguration(
                test_id=f"stress_{users}u_{spawn_rate}s",
                test_type=TestType.STRESS,
                concurrent_users=users,
                spawn_rate=spawn_rate,
                duration=duration,
                network_condition=NetworkCondition.WIFI,
                api_endpoints=self.api_endpoints,
                target_url=self.base_url
            )
            configs.append(config)
        
        return configs
    
    def generate_all_configs(self) -> List[TestConfiguration]:
        """Generate all test configurations"""
        configs = []
        configs.extend(self.generate_api_test_configs())
        configs.extend(self.generate_frontend_test_configs())
        configs.extend(self.generate_load_test_configs())
        configs.extend(self.generate_stress_test_configs())
        return configs
    
    def save_configs(self, configs: List[TestConfiguration], filename: str):
        """Save configurations to JSON file"""
        config_dicts = [config.to_dict() for config in configs]
        
        with open(filename, 'w') as f:
            json.dump(config_dicts, f, indent=2)
        
        print(f"Saved {len(configs)} test configurations to {filename}")
    
    def load_configs(self, filename: str) -> List[TestConfiguration]:
        """Load configurations from JSON file"""
        with open(filename, 'r') as f:
            config_dicts = json.load(f)
        
        configs = []
        for config_dict in config_dicts:
            config = TestConfiguration(
                test_id=config_dict['test_id'],
                test_type=TestType(config_dict['test_type']),
                concurrent_users=config_dict['concurrent_users'],
                spawn_rate=config_dict['spawn_rate'],
                duration=config_dict['duration'],
                network_condition=NetworkCondition(config_dict['network_condition']),
                browser=BrowserType(config_dict['browser']) if config_dict.get('browser') else None,
                api_endpoints=config_dict.get('api_endpoints'),
                custom_headers=config_dict.get('custom_headers'),
                payload_size=config_dict.get('payload_size'),
                target_url=config_dict.get('target_url')
            )
            configs.append(config)
        
        return configs

def main():
    """Example usage of the test parameterizer"""
    parameterizer = TestParameterizer()
    
    # Generate all configurations
    configs = parameterizer.generate_all_configs()
    
    # Save to file
    parameterizer.save_configs(configs, "test_configurations.json")
    
    # Print summary
    print(f"\nGenerated {len(configs)} test configurations:")
    print(f"- API tests: {len([c for c in configs if c.test_type == TestType.API])}")
    print(f"- Frontend tests: {len([c for c in configs if c.test_type == TestType.FRONTEND])}")
    print(f"- Load tests: {len([c for c in configs if c.test_type == TestType.LOAD])}")
    print(f"- Stress tests: {len([c for c in configs if c.test_type == TestType.STRESS])}")
    
    # Show example configurations
    print("\nExample API test configuration:")
    api_config = next(c for c in configs if c.test_type == TestType.API)
    print(json.dumps(api_config.to_dict(), indent=2))
    
    print("\nExample Frontend test configuration:")
    frontend_config = next(c for c in configs if c.test_type == TestType.FRONTEND)
    print(json.dumps(frontend_config.to_dict(), indent=2))

if __name__ == "__main__":
    main()
