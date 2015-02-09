from unittest import TestSuite, TestLoader, TestResult


def main():
    modules = ['test_360', 'test_dependencies', 'test_optional_dependencies', 'test_simple']
    simple_suite = TestSuite()
    loader = TestLoader()
    result = TestResult()
    for module in modules:
        suite = loader.loadTestsFromName(module)
        simple_suite.addTest(suite)

    print
    print 'Running simple test suite...'
    print

    simple_suite.run(result)

    print
    print 'Ran {0} tests.'.format(result.testsRun)
    print

    if len(result.errors) > 0:
        print '#########################################################'
        print 'There are {0} errors. See below for tracebacks:'.format(len(result.errors))
        print '#########################################################'
        print
        for error in result.errors:
            print error[1]
        print
        print '#########################################################'
        print 'There are {0} errors. See above for tracebacks.'.format(len(result.errors))
        print '#########################################################'
    else:
        print 'All tests passed.'
    print

if __name__ == '__main__':
    main()