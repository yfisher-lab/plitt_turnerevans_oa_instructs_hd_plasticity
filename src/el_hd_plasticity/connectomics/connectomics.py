"""Hemibrain connectome helpers: neuPrint client and the ellipsoid body (EB) coordinate system.

:class:`EBCoordinateSystem` re-expresses hemibrain coordinates in a frame aligned to the EB
(principal axes of the EB mesh) and provides ring-phase / radius / cross-section coordinates plus
normalized versions of them, so synapse and skeleton positions can be compared across the ring.
"""

import navis.interfaces.neuprint as neu
import neuprint as npt
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA

from . import params


def _flip_yz(xyz):
    """Flip the y and z axes of (N, 3) EB-basis coordinates (copy), the convention of all EB-frame figures."""
    xyz = np.array(xyz, dtype=float)
    xyz[:, 1:] *= -1
    return xyz


def npt_client(webpage='neuprint.janelia.org', dataset=params.NPT_DATASET, token=params.NPT_TOKEN):
    """Create a neuPrint client (also registers it as the default client for neuprint/navis calls).

    Args:
        webpage (str): The URL of the neuprint server.
        dataset (str): The name of the dataset to use.
        token (str): The authentication token for the neuprint server.

    Returns:
        npt.Client: A neuprint client object.
    """
    return npt.Client(webpage, dataset=dataset, token=token)



class EBCoordinateSystem:
    """Manage coordinate system transformations for the ellipsoid body (EB) in the Drosophila brain.
    
    Provides methods to convert between different coordinate systems (Cartesian, polar/cylindrical)
    and handle both the main EB coordinate system and cross-sectional coordinates.
    """

    def __init__(self):
        """Fetch the EB mesh from neuPrint (needs a registered client, see :func:`npt_client`) and build
        the coordinate transforms and radius scalers."""

        self.orig_eb_mesh = neu.fetch_roi("EB")
        self.orig_origin = self.orig_eb_mesh.vertices.mean(axis=0)
        self.eb_basis = PCA(n_components=3).fit(self.orig_eb_mesh.vertices).components_


        self.eb_vertices = None
        self.eb_vertices_circ = {'phase':None, 'radius':None, 'radius_scaled':None}
        self._get_eb_mesh_coordinates()

        self._radius_min_lookup = None
        self._radius_max_lookup = None
        self._build_eb_rad_scaler()
        self._scaled_eb_radius()

        self.eb_vertices_xsec = {'radius':None, 'phase':None, 'scaled_radius': None}
        self._xsec_radius_lookup = None
        self._xsec_z_lookup = None
        self._build_xsec_lookup()
        self._get_eb_xsec_coords()

        self._xsec_rad_spline = None
        self._fit_xsec_rad_spline()
        self._scale_eb_xsec_rad()

        
    
    def change_of_basis(self, x):
        """Convert coordinates from Neuprint coordinate system to EB basis.
        
        Args:
            x: (N, 3) array of coordinates in Neuprint coordinate system (x, y, z).
            
        Returns:
            (N, 3) array of coordinates transformed to EB basis.
        """
        return (x-self.orig_origin) @ self.eb_basis.T
    
    def to_eb_frame(self, x):
        """Convert Neuprint coordinates to the EB frame used in the figures: EB basis with y and z flipped.

        Args:
            x: (N, 3) array of coordinates in Neuprint coordinate system (x, y, z).

        Returns:
            (N, 3) array in the EB frame.
        """
        return _flip_yz(self.change_of_basis(x))

    def fetch_roi(self, roi):
        """Mesh of a hemibrain ROI (e.g. 'EB', 'PB') with its vertices in the EB frame."""
        mesh = neu.fetch_roi(roi)
        mesh.vertices = self.to_eb_frame(mesh.vertices)
        return mesh

    def fetch_skeleton(self, body_id):
        """navis skeleton of a neuron (see :func:`navis.interfaces.neuprint.fetch_skeletons`) in the EB frame."""
        return self.skeleton_to_eb_coords(neu.fetch_skeletons(body_id))

    def fetch_skeletons(self, body_ids):
        """navis skeletons of several neurons in the EB frame, fetched in one query.

        Args:
            body_ids: Body IDs of the neurons.

        Returns:
            navis NeuronList with one skeleton per body ID.
        """
        skeletons = neu.fetch_skeletons(list(body_ids))
        for skeleton in skeletons:
            nodes = skeleton.nodes.copy()
            nodes.loc[:, ['x', 'y', 'z']] = self.to_eb_frame(nodes.loc[:, ['x', 'y', 'z']].values).astype(np.float32)
            skeleton.nodes = nodes
        return skeletons

    @staticmethod
    def cart3d_to_polar2d(x):
        """Convert 3D Cartesian coordinates to 2D polar coordinates.
        
        Assumes data is centered at origin. Extracts phase and radius from x,y components.
        
        Args:
            x: (N, 3) array of 3D coordinates (x, y, z).
            
        Returns:
            tuple: (phase, radius) where phase is the angle in radians and radius is the distance from origin.
        """
        phase = np.arctan2(x[:,1], x[:,0])
        radius = np.sqrt(x[:,0]**2 + x[:,1]**2)
        return phase, radius
    
    def _get_eb_mesh_coordinates(self):
        """Compute EB mesh coordinates in the transformed coordinate system.
        
        Transforms mesh vertices to EB basis and converts to polar coordinates.
        """
        self.eb_vertices = self.change_of_basis(self.orig_eb_mesh.vertices)
        phase, radius = self.cart3d_to_polar2d(self.eb_vertices)
        self.eb_vertices_circ['phase'], self.eb_vertices_circ['radius'] = phase, radius

    def _build_eb_rad_scaler(self):
        """Build interpolation functions for minimum and maximum radius as a function of phase.
        
        Bins the EB mesh by phase angle and computes the minimum and maximum radius in each bin.
        Creates cubic spline interpolations for smooth scaling across all phases.
        """
        phase, radius = self.eb_vertices_circ['phase'], self.eb_vertices_circ['radius']
        
        bins = np.linspace(-np.pi, np.pi, num=int(360/2) + 1)
        phase_bin_inds = np.digitize(phase, bins) 
        radius_min = [np.amin(radius[phase_bin_inds == i]) for i in range(1, len(bins))]
        radius_max = [np.amax(radius[phase_bin_inds == i]) for i in range(1, len(bins))]

        x = np.concatenate([bins[:-1]-2*np.pi, bins[:-1], bins[:-1]+2*np.pi])
        self._radius_min_lookup = sp.interpolate.interp1d(x, np.concatenate([radius_min for i in range(3)]), kind='cubic')
        self._radius_max_lookup = sp.interpolate.interp1d(x, np.concatenate([radius_max for i in range(3)]), kind='cubic')

        

    def scale_radius(self, phase, radius):
        """Normalize radius to a standard scale based on phase.
        
        Scales radius values to be between 0 and 1 relative to the min/max radius at each phase,
        then applies a linear transformation to the range [0.2/1.2, 1.2/1.2].
        
        Args:
            phase: Phase angle values.
            radius: Radius values to scale.
            
        Returns:
            Scaled radius values.
        """
        return ((radius - self._radius_min_lookup(phase))/(self._radius_max_lookup(phase)-self._radius_min_lookup(phase)) + .2)/1.2
    

    def _scaled_eb_radius(self):
        """Scale the EB mesh radius values and store the results.
        
        Applies radius scaling to all EB mesh vertices and stores the scaled values.
        """
        phase, radius = self.eb_vertices_circ['phase'], self.eb_vertices_circ['radius']
        self.eb_vertices_circ['radius_scaled'] = self.scale_radius(phase, radius)
        

    def get_circ_coordinates(self, eb_x):
        """Get circular (polar) coordinates from EB Cartesian coordinates.
        
        Args:
            eb_x: (N, 3) coordinates after applying change of basis.
            
        Returns:
            tuple: (phase, radius) in polar coordinates.
        """
        
        return self.cart3d_to_polar2d(eb_x)
    
    def _build_xsec_lookup(self):
        """Build interpolation functions for cross-sectional radius and height.
        
        Bins the EB mesh by phase and computes mean radius and height in each bin.
        Creates cubic spline interpolations for smooth lookup across all phases.
        """
        radius, phase, height = self.eb_vertices_circ['radius'], self.eb_vertices_circ['phase'], self.eb_vertices[:,2]

        bins = np.linspace(-np.pi, np.pi, num=int(360/2) + 1)
        phase_bin_inds = np.digitize(phase, bins) 

        r_com, h_com = [], []
        for i in range(1, len(bins)):
            rads = radius[phase_bin_inds==i]
            h = height[phase_bin_inds==i]
            
            r_com.append(rads.mean())
            h_com.append(h.mean())

        x = np.concatenate([bins[:-1]-2*np.pi, bins[:-1], bins[:-1]+2*np.pi])
        self._xsec_radius_lookup = sp.interpolate.interp1d(x, np.concatenate([r_com for i in range(3)]), kind='cubic')
        self._xsec_z_lookup = sp.interpolate.interp1d(x, np.concatenate([h_com for i in range(3)]), kind='cubic')

    def get_xsec_coords(self, eb_phase, radius, z):
        """Get cross-sectional coordinates within the EB structure.
        
        Converts 3D coordinates to a 2D cross-sectional representation by computing
        offset from the main circular axis.
        
        Args:
            eb_phase: Phase along main circular axis of ring.
            radius: Radius along main circular axis.
            z: Height in EB Cartesian basis.
            
        Returns:
            tuple: (xsec_phase, xsec_radius) cross-sectional polar coordinates.
        """

        x = radius - self._xsec_radius_lookup(eb_phase)
        y = z - self._xsec_z_lookup(eb_phase)
        xy = np.zeros([x.shape[0],2])
        xy[:,0], xy[:,1] = x, y

        return self.cart3d_to_polar2d(xy)
    
    def _get_eb_xsec_coords(self):
        """Compute and store cross-sectional coordinates for all EB mesh vertices."""
        self.eb_vertices_xsec['phase'], self.eb_vertices_xsec['radius'] = self.get_xsec_coords(
                                                                                self.eb_vertices_circ['phase'], 
                                                                                self.eb_vertices_circ['radius'], 
                                                                                self.eb_vertices[:,2]
                                                                                )
        
    def _fit_xsec_rad_spline(self):
        """Fit a 2D spline to cross-sectional radius as a function of both phase coordinates.
        
        Uses least squares bivariate spline fitting to model how cross-sectional radius
        varies with both the main EB phase and the cross-sectional phase.
        """
        knots = np.linspace(-np.pi,np.pi, num = 20)
        eb_phase, xsec_phase, xsec_rad = self.eb_vertices_circ['phase'], self.eb_vertices_xsec['phase'], self.eb_vertices_xsec['radius']
        self._xsec_rad_spline = sp.interpolate.LSQBivariateSpline(eb_phase, xsec_phase, xsec_rad, knots, knots)

    def scale_xsec_rad(self, eb_phase, xsec_phase, xsec_rad):
        """Scale cross-sectional radius by the fitted spline model.
        
        Normalizes cross-sectional radius values by dividing by the expected value
        from the fitted spline at the given phase coordinates.
        
        Args:
            eb_phase: Phase along main circular axis.
            xsec_phase: Phase within cross-section.
            xsec_rad: Cross-sectional radius to scale.
            
        Returns:
            Scaled cross-sectional radius values.
        """

        return xsec_rad/self._xsec_rad_spline.ev(eb_phase, xsec_phase)

    def _scale_eb_xsec_rad(self):
        """Scale and store the cross-sectional radius for all EB mesh vertices.
        
        Applies cross-sectional radius scaling to all EB mesh vertices and stores
        the scaled values in the eb_vertices_xsec dictionary.
        """
        self.eb_vertices_xsec['radius_scaled'] = self.scale_xsec_rad(
                                                        self.eb_vertices_circ['phase'], 
                                                        self.eb_vertices_xsec['phase'],
                                                        self.eb_vertices_xsec['radius']
                                                        )

    def ring_outline(self, n=361):
        """Inner and outer outline of the EB mesh in the xy plane of the EB frame (as used by :meth:`to_eb_frame`).

        Returns:
            Tuple of two (n, 2) arrays (inner and outer edge), each running once around the ring.
        """
        phase = np.linspace(-np.pi, np.pi, num=n)
        lookup = -phase                # the lookups are in the EB basis; the EB frame flips y, i.e. negates the phase
        edges = []
        for radius in (self._radius_min_lookup(lookup), self._radius_max_lookup(lookup)):
            edges.append(np.column_stack((radius * np.cos(phase), radius * np.sin(phase))))
        return tuple(edges)

    def skeleton_to_eb_coords(self, navis_skeleton):
        """Express a navis skeleton's node coordinates in the EB basis (y and z flipped).

        The flips match :func:`add_eb_coords` so skeletons and synapses share one frame.

        Args:
            navis_skeleton: navis NeuronList holding one skeleton in Neuprint coordinates, as
                returned by ``navis.interfaces.neuprint.fetch_skeletons``.

        Returns:
            The same NeuronList, with its node table replaced by one in EB coordinates (float32).
        """
        nodes = navis_skeleton.nodes.copy()
        nodes.loc[:, ['x', 'y', 'z']] = self.to_eb_frame(nodes.loc[:, ['x', 'y', 'z']].values).astype(np.float32)
        navis_skeleton.set_neuron_attributes(nodes, name='nodes')
        return navis_skeleton


def add_eb_coords(syn_df, ebcoords, type='post'):
    """Add EB-frame coordinates of synapses to a synapse table, in place.

    Adds columns x_eb, y_eb, z_eb (EB basis, y and z flipped), phase, radius, radius_scaled
    (ring coordinates), xsec_phase, xsec_radius, xsec_radius_scaled (cross-section polar
    coordinates) and xsec_x/xsec_y (+ ``_scaled`` versions) in Cartesian form.

    Args:
        syn_df: DataFrame from ``neuprint.fetch_synapse_connections`` with columns
            x_pre/y_pre/z_pre and x_post/y_post/z_post, or from ``neuprint.fetch_synapses`` with columns x/y/z
            (``type=None``).
        ebcoords: :class:`EBCoordinateSystem`.
        type: Which side of the synapse to locate, 'pre' or 'post'; None to use the plain x, y, z columns.

    Returns:
        `syn_df` (modified in place).
    """
    suffix = '' if type is None else f'_{type}'
    syn_df.loc[:, ['x_eb', 'y_eb', 'z_eb']] = _flip_yz(ebcoords.change_of_basis(
        syn_df.loc[:, [f'x{suffix}', f'y{suffix}', f'z{suffix}']].values))
    phase, radius = ebcoords.get_circ_coordinates(syn_df.loc[:, ['x_eb', 'y_eb', 'z_eb']].values)
    syn_df['phase'], syn_df['radius'] = phase, radius
    syn_df['radius_scaled'] = ebcoords.scale_radius(syn_df['phase'], syn_df['radius'])
    syn_df['xsec_phase'], syn_df['xsec_radius'] = ebcoords.get_xsec_coords(syn_df['phase'], syn_df['radius'], syn_df['z_eb'])
    syn_df['xsec_radius_scaled'] = ebcoords.scale_xsec_rad(syn_df['phase'],
                                                           syn_df['xsec_phase'],
                                                           syn_df['xsec_radius'])

    syn_df['xsec_x'] = syn_df['xsec_radius'] * np.cos(syn_df['xsec_phase'])
    syn_df['xsec_y'] = syn_df['xsec_radius'] * np.sin(syn_df['xsec_phase'])
    syn_df['xsec_x_scaled'] = syn_df['xsec_radius_scaled'] * np.cos(syn_df['xsec_phase'])
    syn_df['xsec_y_scaled'] = syn_df['xsec_radius_scaled'] * np.sin(syn_df['xsec_phase'])
    return syn_df


def peeled_convex_hull(syn_points, n=20):
    """Convex hull of a point cloud after peeling off the outermost hull `n` times.

    Repeatedly removes the points on the convex hull, which makes the final hull robust to a
    few outlying synapses.

    Args:
        syn_points: (N, 2) array of point coordinates.
        n: Number of times to peel off the convex hull.

    Returns:
        (M, 2) array with the vertices of the convex hull of the remaining points.
    """
    syn_points = syn_points.copy()
    for _ in range(n):
        syn_hull = sp.spatial.ConvexHull(syn_points)
        syn_points = np.delete(syn_points, syn_hull.vertices, axis=0)
    return syn_points[sp.spatial.ConvexHull(syn_points).vertices, :]


def fetch_eb_synapses(ebcoords, body_ids, side='pre', roi='EB'):
    """Synapses of neurons in `roi` with their EB-frame coordinates (see :func:`add_eb_coords`).

    Args:
        ebcoords: :class:`EBCoordinateSystem`.
        body_ids: Body ID(s) of the neurons.
        side: 'pre' or 'post' synapses.
        roi: ROI the synapses are in.
    """
    syn = neu.fetch_synapses(body_ids, neu.SynapseCriteria(rois=[roi], type=side))
    return add_eb_coords(syn, ebcoords, type=None)


def synapse_phase(syn_xy, n=10):
    """Ring phase of a neuron from the outline of its synapses.

    The outline is the peeled convex hull of the synapse positions in the EB-frame xy plane (see
    :func:`peeled_convex_hull`); the phase is the circular mean of the angles of its vertices.

    Args:
        syn_xy: (N, 2) array of synapse x, y coordinates in the EB frame.
        n: Number of times the convex hull is peeled.

    Returns:
        Tuple of (hull vertices, phase in radians).
    """
    hull = peeled_convex_hull(syn_xy, n=n)
    phase = np.angle(np.exp(1j * np.arctan2(hull[:, 1], hull[:, 0])).mean())
    return hull, phase


def phase_width(phase, low=5, high=95):
    """Width (radians) of a distribution of ring phases: the span between two percentiles around its circular mean.

    Args:
        phase: Array of phases in radians (e.g. the ``phase`` column of :func:`add_eb_coords`).
        low, high: Percentiles bounding the width.
    """
    centered = np.angle(np.exp(1j * (phase - np.angle(np.exp(1j * phase).mean()))))
    return np.angle(np.exp(1j * (np.percentile(centered, high) - np.percentile(centered, low))))
