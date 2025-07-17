# First, import the modules that are assumed to be always available
from .sokoban import SokobanEnv, SokobanEnvConfig, SokobanService, SokobanServiceConfig
from .frozenlake import FrozenLakeEnv, FrozenLakeEnvConfig, FrozenLakeService, FrozenLakeServiceConfig

REGISTERED_ENV = {
    "sokoban": {
        "env_cls": SokobanEnv,
        "config_cls": SokobanEnvConfig,
        "service_cls": SokobanService,
        "service_config_cls": SokobanServiceConfig
    },
    "frozenlake": {
        "env_cls": FrozenLakeEnv,
        "config_cls": FrozenLakeEnvConfig,
        "service_cls": FrozenLakeService,
        "service_config_cls": FrozenLakeServiceConfig
    }
}

try:
    from .navigation import NavigationEnv, NavigationEnvConfig, NavigationServiceConfig, NavigationService
    REGISTERED_ENV["navigation"] = {
        "env_cls": NavigationEnv,
        "config_cls": NavigationEnvConfig,
        "service_cls": NavigationService,
        "service_config_cls": NavigationServiceConfig
    }
except ImportError:
    pass

try:
    from .svg import SVGEnv, SvgEnvConfig, SVGService, SVGServiceConfig
    REGISTERED_ENV["svg"] = {
        "env_cls": SVGEnv,
        "config_cls": SvgEnvConfig,
        "service_cls": SVGService,
        "service_config_cls": SVGServiceConfig
    }
except ImportError:
    pass

try:
    from .primitive_skill import PrimitiveSkillEnv, PrimitiveSkillEnvConfig, PrimitiveSkillService, PrimitiveSkillServiceConfig
    REGISTERED_ENV["primitive_skill"] = {
        "env_cls": PrimitiveSkillEnv,
        "config_cls": PrimitiveSkillEnvConfig,
        "service_cls": PrimitiveSkillService,
        "service_config_cls": PrimitiveSkillServiceConfig
    }
except ImportError:
    pass


try:
    from .alfworld import ALFWorldEnv, ALFWorldEnvConfig, ALFWorldService, ALFWorldServiceConfig
    REGISTERED_ENV["alfworld"] = {
        "env_cls": ALFWorldEnv,
        "config_cls": ALFWorldEnvConfig,
        "service_cls": ALFWorldService,
        "service_config_cls": ALFWorldServiceConfig
    }
except ImportError:
    pass

try:
    from .spatial import SpatialGym, SpatialGymConfig, SpatialGymService, SpatialGymServiceConfig
    REGISTERED_ENV["spatial"] = {
        "env_cls": SpatialGym,
        "config_cls": SpatialGymConfig,
        "service_cls": SpatialGymService,
        "service_config_cls": SpatialGymServiceConfig
    }
except ImportError:
    pass