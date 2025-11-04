"""Docker container management utilities"""
import subprocess
import docker
from typing import Dict, List, Optional
import time


class DockerManager:
    """Manage Docker containers and compose services"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"Warning: Could not connect to Docker daemon: {e}")
            self.client = None
    
    def check_container_status(self, container_name: str) -> Dict:
        """
        Check if a container is running and healthy
        
        Returns:
            dict: {
                'exists': bool,
                'running': bool,
                'status': str,
                'health': str
            }
        """
        if not self.client:
            return {'exists': False, 'running': False, 'status': 'Docker unavailable', 'health': 'unknown'}
        
        try:
            container = self.client.containers.get(container_name)
            health = container.attrs.get('State', {}).get('Health', {}).get('Status', 'unknown')
            return {
                'exists': True,
                'running': container.status == 'running',
                'status': container.status,
                'health': health
            }
        except docker.errors.NotFound:
            return {'exists': False, 'running': False, 'status': 'not found', 'health': 'unknown'}
        except Exception as e:
            return {'exists': False, 'running': False, 'status': f'error: {str(e)}', 'health': 'unknown'}
    
    def get_all_services_status(self, containers: Dict[str, str]) -> Dict[str, Dict]:
        """Get status of all monitored containers"""
        status = {}
        for service_name, container_name in containers.items():
            status[service_name] = self.check_container_status(container_name)
        return status
    
    def start_services(self, compose_file: str) -> tuple[bool, str]:
        """
        Start services defined in docker-compose file
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            cmd = f"cd {self.project_root} && docker-compose -f {compose_file} up -d"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return True, f"✅ Services started successfully"
            else:
                return False, f"❌ Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "❌ Timeout: Service startup took too long"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def stop_services(self, compose_file: str) -> tuple[bool, str]:
        """
        Stop services defined in docker-compose file
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            cmd = f"cd {self.project_root} && docker-compose -f {compose_file} down"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return True, f"✅ Services stopped successfully"
            else:
                return False, f"❌ Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "❌ Timeout: Service shutdown took too long"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def stop_all_services(self, compose_files: List[str]) -> tuple[bool, str]:
        """Stop all services across multiple compose files"""
        messages = []
        all_success = True
        
        for compose_file in compose_files:
            success, message = self.stop_services(compose_file)
            messages.append(f"{compose_file}: {message}")
            if not success:
                all_success = False
        
        return all_success, "\n".join(messages)
    
    def get_container_logs(self, container_name: str, lines: int = 50) -> str:
        """Get last N lines of container logs"""
        if not self.client:
            return "Docker unavailable"
        
        try:
            container = self.client.containers.get(container_name)
            logs = container.logs(tail=lines, timestamps=True).decode('utf-8')
            return logs
        except docker.errors.NotFound:
            return f"Container '{container_name}' not found"
        except Exception as e:
            return f"Error getting logs: {str(e)}"
    
    def get_container_stats(self, container_name: str) -> Optional[Dict]:
        """Get container resource usage statistics"""
        if not self.client:
            return None
        
        try:
            container = self.client.containers.get(container_name)
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
            
            # Calculate memory usage
            mem_usage = stats['memory_stats'].get('usage', 0)
            mem_limit = stats['memory_stats'].get('limit', 1)
            mem_percent = (mem_usage / mem_limit) * 100.0 if mem_limit > 0 else 0.0
            
            return {
                'cpu_percent': round(cpu_percent, 2),
                'memory_usage_mb': round(mem_usage / (1024 * 1024), 2),
                'memory_limit_mb': round(mem_limit / (1024 * 1024), 2),
                'memory_percent': round(mem_percent, 2)
            }
        except Exception as e:
            print(f"Error getting stats for {container_name}: {e}")
            return None
    
    def restart_container(self, container_name: str) -> tuple[bool, str]:
        """Restart a specific container"""
        if not self.client:
            return False, "Docker unavailable"
        
        try:
            container = self.client.containers.get(container_name)
            container.restart(timeout=30)
            return True, f"✅ Container '{container_name}' restarted successfully"
        except docker.errors.NotFound:
            return False, f"❌ Container '{container_name}' not found"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def restart_services(self, compose_file: str) -> tuple[bool, str]:
        """
        Restart services defined in docker-compose file
        Forces recreation of containers even if they're already running
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            cmd = f"cd {self.project_root} && docker-compose -f {compose_file} up -d --force-recreate"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return True, f"✅ Services restarted successfully"
            else:
                return False, f"❌ Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "❌ Timeout: Service restart took too long"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"

