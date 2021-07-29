.. _utils_ind-index:

=========
Utils InD
=========


**Generating CR scenarios from the inD dataset**

1. Setup of inD data source:

   -  Every OS: copy/link recording data (from inD Dataset) in ``data``
      in root of this project

2. Install requirements for this project (Anaconda is recommended)
3. Make sure this project is properly installed.
4. Change into the ``commonroad_rl/utils_ind`` directory
5. If meta scenarios have not been created yet

   -  install the
      `commonroad-map-tool <https://gitlab.lrz.de/cps/commonroad-map-tool>`__
      (for support of traffic signs and stop lines, install
      `feature\_osm\_convert\_trafficsigns <https://gitlab.lrz.de/cps/commonroad-map-tool/-/tree/feature_osm_convert_traffic_signs>`__)
      or execute ``bash scripts/install_ind_convert.sh`` from the root
      of this project
   -  convert the osm files ``bash   bash convert_osm.sh``

6. Run script (``--help`` for options)
   ``bash    python -m commonroad_rl.utils_ind.ind_to_cr --help``

Typically you would want to run something like the following. It will
convert the recordings 9, 20 and 33 to common road format and output 20
smaller planning problems for each recording, storing them in the
directory ``../../pickles`` in pickle format.

::

    python -m commonroad_rl.utils_ind.ind_to_cr -t 0 9 20 32 -n 20 -o ../../pickles  --multiprocessing -v -p

**Generating gifs of the resulting scenarios**

Note that this process requires properly installed
`ffmpeg <https://ffmpeg.org/>`__

-  run ``visualize_cr.py`` with the ``-o`` option to fill a directory
   with animation frames
-  cd there
-  ``ffmpeg -i %03d.png -vf palettegen palette.png`` (or however many
   digits your images have)
-  ``ffmpeg -i %03d.png -i palette.png -lavfi "paletteuse" -y output.gif``

**Notes on Cartographic data**

* Conversion Lanelet2 -> CR

The original intersection lanelets were supplied in
`Lanelet2 <https://github.com/fzi-forschungszentrum-informatik/Lanelet2>`__
format. To convert it to CommonRoad lanelets, the
`commonroad-map-tool <https://gitlab.lrz.de/cps/commonroad-map-tool/-/tree/develop/>`__
may be used.

.. code:: bash

    osm-convert PATH/TO/OSMMAPS/OSMMAP.osm -o LANELETFILE.xml 

The results may not be perfect, depending on how clean the Lanele2 data
was originally, so I recommend to run ``tools/validate_cr.py`` after
conversion. Note that without any vehicles and planning problem, related
warnings may be output but they may be ignored.

Since the coordinates are originally stored in UTM coordinates (which is
usually not suitable for useful plotting axes) they have been edited by
subtracting the UTM cooridinates associated with them in the
recordingMeta files. This can be done explicitely by

.. code:: bash

    python tools/translate_intersections.py inD_LaneletMaps/convert_tinkered

All of the above, excluding validation, is included in
``osm-convert.sh``.

* Fixing the provided OSM data

Lanelet2 data may be edited and sometimes requires some manual fixing
before conversion.

1. remove sidewalks, pedestrian routes and unnecessary walls
2. make sure all paths along one consecutive logical lanelets are
   pointing in the same direction -> if not, select path in JOSM, select
   "reverse path"
3. make sure all lanelets that are supposed to split/join from another
   lanelet do connect to the (same) endpoints of that other lanelet (the
   correct way of doing this is documented in the `CommonRoad
   Documentation <https://gitlab.lrz.de/tum-cps/commonroad-scenarios/blob/master/documentation/XML_commonRoad_2020a.pdf>`__
4. make sure that the opposing edges of a lanelet have the same number
   of nodes (this is a requirement for valid CR lanelets. the conversion
   tool may try to fix this manually but often causes invalid
   geometries.)
5. convert to lanelet with the
   ```commonroad-map-tool`` <https://gitlab.lrz.de/cps/commonroad-map-tool/-/tree/develop/>`__
   script ``osm-convert``
6. run
   ```xml-scenario-checker`` <https://gitlab.lrz.de/cps/xml-scenario-checker/-/tree/feature_issues_in_network_checker/>`__
   ``lanelet-network-checker/run_vertices_repair.py``
7. run the logical and geometrical checks of aforementioned package
8. run the tool ``validate_cr.py``. You can ignore errors related to
   missing vehicle, but adjacency and polygon errors should not occur.
9. reiterate and fix errors

**Notes on Recordings / Trajectories**

* Generation of planning problems from vehicles


The inD database files store tracks over an immense frame of time. It is
hence to be considered to make use of one recording several times

-  Choose one random vehicle as planning problem.
-  Cut time frame to frames where this vehicle is visible.

Since the time frame of some vehicles is very large, a random subwindow
will be chosen for the vehicle. Future work may enable choosing several
(non-overlapping) subwindows of the vehicles active frames instead.

* Documentation on traffic signs and lights

Since traffic lights mess up vehicle behaviour if the temporal light
state is unknown here is a list of intersections in the dataset
featuring traffic lights. As described in the
`inD-paper <https://arxiv.org/abs/1911.07602>`__ all intersections are
traffic light free though.

Other traffic signs (such as give way signs) are just as important for
correct behaviour prediction so OSM data and aerial data was searched
for signs of them (ie. shadows) Some were already contained in the
supplied data. In general the codes used in the inD dataset are those of
the defining
`Verkehrszeichenkatalog <http://www.vzkat.de/2017/VzKat.htm>`__
[german]. However, a nice overview may be found on
`Wikipedia <https://de.wikipedia.org/wiki/Bildtafel_der_Verkehrszeichen_in_der_Bundesrepublik_Deutschland_von_2013_bis_2017>`__.

+-------------------+-----------------+------------------------+------------------+---------------------------------------------------------------------------------------------------+
| File name         | type            | position               | source           | Notes                                                                                             |
+===================+=================+========================+==================+===================================================================================================+
| Bendplatz.osm     | various         |                        | inD dataset      |                                                                                                   |
+-------------------+-----------------+------------------------+------------------+---------------------------------------------------------------------------------------------------+
| aseag.osm         | give way sign   | 50.7853132 61308116    | aerial imagery   | neuköllner straße (top) entering into charlottenburger alle (bottom) on left side                 |
+-------------------+-----------------+------------------------+------------------+---------------------------------------------------------------------------------------------------+
| heckstrasse.osm   | stop sign       | 50.7788996 6.1654376   | OSM              | this is in the middle of the road. heckstrase (top right street) entering into von-coels-straße   |
+-------------------+-----------------+------------------------+------------------+---------------------------------------------------------------------------------------------------+
| heckstrasse.osm   | various         |                        | inD dataset      |                                                                                                   |
+-------------------+-----------------+------------------------+------------------+---------------------------------------------------------------------------------------------------+

Complementary information on the right of way situation can also be
found in the `inD-paper <https://arxiv.org/abs/1911.07602>`__ (Section:
the dataset at a glance). Virtual and actual traffic signs were added to
the modified Lanelet2 data where possible to be integrated in the
commonraod scenarios.

* A note on right of way relations in CommonRoad

TL;DR The format of traffic signs in commonroad wil make determining
right of way/yield relations in the gym environment troublesome

Digging into the commonroad scenarios, you will find out that there is
no way to store "yield to" and "right of way over" relations in
CommonRoad. The only way to store some priority information is to place
priority lane and yield signs and link them to the lanelet which they
directly affect (i.e. this lanelet has to yield. To whom? Not saved :)).

For once this means wihtout a traffic sign this relation will be
completely lost. Also this makes it (as far as I see) highly non-trivial
to determine "the opposing side" - i.e. which lanelets will the vehicle
have to yield to or have the priority over. An example on how
non-trivial the guesswork on "what lanelets does this traffic sign apply
to" can be found even inside one (normal?) T-crossing there are several
"yield for" relations that would contradict each other if extended too
far.

This is why currently, priority relations are not at all covered by the
Gym environment.


InD Lanelet Maps
================

Osm_tinkered
------------
Contains manually fixed and adjusted OSM files based on the data provided by the inD dataset
| :code:`Benzplatz.osm`
| :code:`Aseag.osm`
| :code:`Frankenberg.osm`
| :code:`Heckstrasse.osm`

Metadata for Intersections
--------------------------
Some example recordings where each intersection is featured plus the
mapping from name to id.

+---------------------+--------------+---------------------+
| Intersection name   | locationID   | example recording   |
+=====================+==============+=====================+
| aseag               | 4            | 00                  |
+---------------------+--------------+---------------------+
| Bendplatz           | 1            | 08                  |
+---------------------+--------------+---------------------+
| frankenberg         | 2            | 19                  |
+---------------------+--------------+---------------------+
| heckstrasse         | 3            | 32                  |
+---------------------+--------------+---------------------+

Tools
=====

Extract Origins
---------------
Extracts UtmOrigins and Speed limits for intersections


Translate Intersections
-----------------------
Translate converted inD intersection lanelet to useful coordinates, translating them into local coordinate system (also better usable by CommonRoad)


Utils
=====

Common
------
.. automodule:: commonroad_rl.utils_ind.utils.common
   :members:
   :member-order: bysource

Csv to planning problem
-----------------------
.. automodule:: commonroad_rl.utils_ind.utils.csv_to_planning_problem
   :members:
   :member-order: bysource

Dynamic Obstacle to planning problem
------------------------------------
.. automodule:: commonroad_rl.utils_ind.utils.dyn_obs_to_planning_problem
   :members:
   :member-order: bysource

Tracks import
-------------
.. automodule:: commonroad_rl.utils_ind.utils.tracks_import
   :members:
   :member-order: bysource

Ind to CommonRoad
=================
.. automodule:: commonroad_rl.utils_ind.ind_to_cr
   :members:
   :member-order: bysource

