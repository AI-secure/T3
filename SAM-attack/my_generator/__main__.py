import logging

from common.QAConfig import QAConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(QAConfig.log_dir, 'generator.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info('123')

