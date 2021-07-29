.. _documentation-index:

====================
Documentation Manual
====================

.. hint::

   This is a hint and it can added with :code:`.. hint::`. The context of the hint has to be indented.

.. Too much inline code can make the information on a page highly
   unreadable. If this is the case, consider using
   :ref:`writing-rest-codeblocks-with-syntax-highlighting`.

.. only:: html

    :Release: |version|
    :Date: |today|

In this section, we will explain how to extend this documentation. We will first explain the structure of the documents and then we will explain the syntax.

Structure
=========

The structure of this documentation is the same as the structure of the commonroad-rl package. If you add a file called :code:`example.py` in the folder 
:code:`gym_commonroad`, the documentation of this file should be located in a file :code:`example.rst` in the :code:`gym_commonroad` folder of the documentation, e.g. :code:`commonroad-rl/commonroad_rl/doc/gym-commonroad`.

If you want to link to other files, you can do this with::

    .. toctree::
            :maxdepth: 1

            example.rst

The ``maxdepth`` parameter signifies how deep the link structure is. Each time the ``toctree`` is used a additional depth is created. 
If ``example.rst`` would also link to other files we would need to adjust the depth accordingly.

Syntax
======

Headers
-------

The headers of a :code:`.rst` file are as follows::

    ===========
    Main Header
    ===========
    Each file should have exactly one main header.

    Header 1.1
    ==========

    Header 1.1.1
    ------------

Code and Highlights
-------------------

In this part, we will explain different ways to highlight and add code to your documentation. If you do not find what you need here you might 
check `this website <https://docs.typo3.org/m/typo3/docs-how-to-document/master/en-us/WritingReST/InlineCode.html>`__.

- inline highlight ``:code:`highlight``` or ````highlight````
- link ```example link <https://example.com>`__``
- large code block can be entered::

    this text is in a code block
    everything after two colons is within that block::

        as long as the lines are inset with a tab an there is a row between the line
        ending with :: and the first line within the code-block

- if you want your code to be highlighted according to your programming language you can use ``.. code-block:: python``

.. code-block:: python

    if a > b:
        pass
    else:
        return 0

Classes
-------

Classes and their members and functions can be automatically documented with the following code ::

    .. automodule:: commonroad_rl.gym_commonroad.commonroad_env

    ``CommonroadEnv`` class
    ^^^^^^^^^^^^^^^^^^^^^^^
    .. autoclass:: CommonroadEnv
       :members:
       :private-members:
       :member-order: bysource

.. automodule:: commonroad_rl.gym_commonroad.commonroad_env

the result of the block above is depicted below.

CommonroadEnv
-------------

``CommonroadEnv`` class
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: CommonroadEnv(self,meta_scenario_path=PATH_PARAMS["meta_scenario"], train_reset_config_path=PATH_PARAMS["train_reset_config"], test_reset_config_path=PATH_PARAMS["test_reset_config"], visualization_path=PATH_PARAMS["visualization"], logging_path=None, test_env=False, play=False, config_file=PATH_PARAMS["configs"], verbose=1, **kwargs)
   :members:
   :private-members:
   :member-order: bysource

.. Test
   ====

   .. toctree::
    :maxdepth: 1

    test.rst


