#Legacy modules go here
import logging
logger = logging.getLogger(__name__)

#tell the end user that this is a legacy module and should not be used in new code
logger.warning("The gigaseal._legacy package is deprecated and should not be used in new code. Please use the new modules in gigaseal.analysis.builtins instead.")