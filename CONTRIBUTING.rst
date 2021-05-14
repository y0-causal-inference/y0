Contributing
============
Contributions, whether big or small, are appreciated! You can get involved by submitting an issue, making a suggestion,
or adding code to the project.

Having a Problem? Submit an Issue.
----------------------------------
1. Check that you have the latest version of :code:`y0`
2. Check that StackOverflow hasn't already solved your problem
3. Go here: https://github.com/y0-causal-inference/y0/issues
4. Check that this issue hasn't been solved
5. Click "new issue"
6. Add a short, but descriptive title
7. Add a full description of the problem, including the code that caused it and any support files related to this code
   so others can reproduce your problem
8. Copy the output and error message you're getting

Have a Question or Suggestion?
------------------------------
Same drill! Submit an issue and we'll have a nice conversation in the thread.

Want to Contribute?
-------------------
1. Get the code. Fork the repository from GitHub using the big green button in the top-right corner of
   https://github.com/y0-causal-inference/y0
2. Clone your directory with

.. code-block:: sh

   $ git clone https://github.com/<YourUsername>/y0

3. Install with :code:`pip`. The flag, :code:`-e`, makes your installation editable, so your changes will be reflected
   automatically in your installation.

.. code-block:: sh

    $ cd y0
    $ python3 -m pip install -e .

4. Make a branch off of develop, then make contributions! This line makes a new branch and checks it out

.. code-block:: sh

    $ git checkout -b feature/<YourFeatureName>

5. This project should be well tested, so write unit tests in the :code:`tests/` directory
6. Check that all tests are passing and code coverage is good with :code:`tox` before committing.

.. code-block:: sh

    $ tox

Pull Requests
~~~~~~~~~~~~~
Once you've got your feature or bugfix finished (or if its in a partially complete state but you want to publish it
for comment), push it to your fork of the repository and open a pull request against the develop branch on GitHub.

Make a descriptive comment about your pull request, perhaps referencing the issue it is meant to fix (something along
the lines of "fixes issue #10" will cause GitHub to automatically link to that issue). The maintainers will review your
pull request and perhaps make comments about it, request changes, or may pull it in to the develop branch! If you need
to make changes to your pull request, simply push more commits to the feature branch in your fork to GitHub and they
will automatically be added to the pull. You do not need to close and reissue your pull request to make changes!

If you spend a while working on your changes, further commits may be made to the main :code:`y0`
repository (called "upstream") before you can make your pull request. In keep your fork up to date with upstream by
pulling the changes--if your fork has diverged too much, it becomes difficult to properly merge pull requests without
conflicts.

To pull in upstream changes:

.. code-block:: sh

    $ git remote add upstream https://github.com/y0-causal-inference/y0
    $ git fetch upstream develop

Check the log to make sure the upstream changes don't affect your work too much:

.. code-block:: sh

    $ git log upstream/develop

Then merge in the new changes:

.. code-block:: sh

    $ git merge upstream/develop

More information about this whole fork-pull-merge process can be found
`here on Github's website <https://help.github.com/articles/fork-a-repo/>`_.
