import unittest
import coverage

cov = coverage.Coverage()
cov.start()

suite = unittest.TestLoader().discover(".", pattern="Test[^.]*.py")
unittest.TextTestRunner(verbosity=2).run(suite)

cov.stop()
cov.report()
